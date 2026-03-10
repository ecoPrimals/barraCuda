// SPDX-License-Identifier: AGPL-3.0-only
//! Inverse 1D Fast Fourier Transform Operation
//!
//! **Purpose**: Transform from frequency domain back to time/spatial domain
//! **Algorithm**: FFT with inverse=true + normalization by 1/N
//!
//! **Mathematical Property**:
//! ```text
//! IFFT(FFT(x)) = x
//! ```
//! This is THE validation test for FFT correctness!

use super::fft_1d::Fft1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::mem::size_of;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalizeParams {
    degree: u32,
    padding: [u32; 3],
}

/// 1D Inverse Complex FFT operation
pub struct Ifft1D {
    input: Tensor,
    degree: u32,
}

impl Ifft1D {
    /// Create a new 1D IFFT operation
    /// # Errors
    /// Returns [`Err`] if input last dimension is not 2, or degree is not a power of 2.
    pub fn new(input: Tensor, degree: u32) -> Result<Self> {
        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "IFFT input must have last dimension = 2 (complex)".to_string(),
            ));
        }

        if degree & (degree - 1) != 0 {
            return Err(BarracudaError::Device(format!(
                "IFFT degree {degree} must be power of 2"
            )));
        }

        Ok(Self { input, degree })
    }

    /// Execute IFFT transformation
    /// Returns time/spatial domain representation (normalized by 1/N).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device().clone();
        let n = self.degree;
        let buffer_bytes = (n as u64) * 2 * size_of::<f32>() as u64;

        // Run forward FFT engine with inverse=true (uses conjugated twiddles via inverse flag)
        let fft = Fft1D::new(self.input, n)?;
        let butterfly_result = fft.execute_internal(true)?;

        // Normalize by 1/N
        let final_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IFFT Final Buffer"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let normalize_params = NormalizeParams {
            degree: n,
            padding: [0; 3],
        };
        let normalize_params_buffer =
            device.create_uniform_buffer("IFFT Normalize Params", &normalize_params);

        let normalize_shader = include_str!("ifft_normalize.wgsl");

        ComputeDispatch::new(&device, "IFFT Normalize")
            .shader(normalize_shader, "main")
            .storage_read(0, butterfly_result.buffer())
            .storage_rw(1, &final_buffer)
            .uniform(2, &normalize_params_buffer)
            .dispatch_1d(n)
            .submit()?;

        Ok(Tensor::from_buffer(
            final_buffer,
            vec![n as usize, 2],
            device,
        ))
    }
}
