// SPDX-License-Identifier: AGPL-3.0-or-later
//! Inverse 1D Fast Fourier Transform Operation
//!
//! **Purpose**: Transform from frequency domain back to time/spatial domain
//! **Algorithm**: FFT with conjugated twiddles + normalization by 1/N
//!
//! **Mathematical Property**:
//! ```text
//! IFFT(FFT(x)) = x
//! ```
//! This is THE validation test for FFT correctness!

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// 1D Inverse Complex FFT operation
pub struct Ifft1D {
    input: Tensor,
    degree: u32,
    twiddle_factors: Vec<f32>,
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

        let _device = input.device();

        // Precompute twiddle factors (CONJUGATED for inverse transform)
        // twiddle[k] = exp(+2πik/N) (note: positive sign for inverse!)
        let mut twiddle_factors = Vec::with_capacity((degree * 2) as usize);
        let pi = std::f32::consts::PI;

        for k in 0..degree {
            let angle = 2.0 * pi * (k as f32) / (degree as f32); // Positive for IFFT!
            let real = angle.cos();
            let imag = angle.sin();
            twiddle_factors.push(real);
            twiddle_factors.push(imag);
        }

        Ok(Self {
            input,
            degree,
            twiddle_factors,
        })
    }

    /// Execute IFFT transformation
    /// Returns time/spatial domain representation.
    /// Output is normalized by 1/N.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let buffer_size = self.degree as u64 * 2 * std::mem::size_of::<f32>() as u64;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IFFT Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let intermediate_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IFFT Intermediate Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IFFT Twiddle Factors"),
                contents: bytemuck::cast_slice(&self.twiddle_factors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let fft_shader = include_str!("fft_1d.wgsl");
        let normalize_shader = include_str!("ifft_normalize.wgsl");

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FftParams {
            degree: u32,
            stage: u32,
            inverse: u32,
            _padding: u32,
        }

        let base_params = FftParams {
            degree: self.degree,
            stage: 0,
            inverse: 1, // IFFT flag
            _padding: 0,
        };

        // Pass 1: Bit-reversal
        let params_buffer = device.create_uniform_buffer("IFFT Params (Bit Reverse)", &base_params);

        ComputeDispatch::new(device, "IFFT Bit Reverse")
            .shader(fft_shader, "bit_reverse")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &intermediate_buffer)
            .storage_read(2, &twiddle_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(self.degree)
            .submit()?;

        // Pass 2-N: Butterfly stages
        let num_stages = (self.degree as f32).log2() as u32;
        let mut current_input = intermediate_buffer;
        let mut current_output = output_buffer;

        for stage in 0..num_stages {
            let stage_params = FftParams {
                stage,
                ..base_params
            };
            let stage_params_buffer = device
                .create_uniform_buffer(&format!("IFFT Params (Stage {stage})"), &stage_params);

            ComputeDispatch::new(device, &format!("IFFT Butterfly Stage {stage}"))
                .shader(fft_shader, "main")
                .storage_read(0, &current_input)
                .storage_rw(1, &current_output)
                .storage_read(2, &twiddle_buffer)
                .uniform(3, &stage_params_buffer)
                .dispatch_1d(self.degree / 2)
                .submit()?;

            std::mem::swap(&mut current_input, &mut current_output);
        }

        // After each stage we swap, so current_input always holds the last written buffer.
        let butterfly_result_buffer = current_input;

        // Pass N+1: Normalize by 1/N
        let final_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IFFT Final Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct NormalizeParams {
            degree: u32,
            _padding: [u32; 3],
        }

        let normalize_params = NormalizeParams {
            degree: self.degree,
            _padding: [0; 3],
        };

        let normalize_params_buffer =
            device.create_uniform_buffer("IFFT Normalize Params", &normalize_params);

        ComputeDispatch::new(device, "IFFT Normalize")
            .shader(normalize_shader, "main")
            .storage_read(0, &butterfly_result_buffer)
            .storage_rw(1, &final_buffer)
            .uniform(2, &normalize_params_buffer)
            .dispatch_1d(self.degree)
            .submit()?;

        Ok(Tensor::from_buffer(
            final_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}
