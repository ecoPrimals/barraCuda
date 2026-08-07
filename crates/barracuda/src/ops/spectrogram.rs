// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectrogram - Power spectrogram computation
//!
//! Computes magnitude squared of STFT.
//! Visualizes frequency content over time.
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

static SHADER_F64: &str = include_str!("../shaders/audio/spectrogram_f64.wgsl");

/// Spectrogram operation
pub struct Spectrogram {
    stft_data: Tensor, // Complex STFT [real, imag, real, imag, ...]
    power: f32,        // 1.0 for magnitude, 2.0 for power
}

impl Spectrogram {
    /// Create a new spectrogram operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(stft_data: Tensor, power: f32) -> Result<Self> {
        let size = stft_data.shape().iter().product::<usize>();
        if size % 2 != 0 {
            return Err(BarracudaError::invalid_input(
                "STFT data must contain even number of elements (complex pairs)",
            ));
        }
        Ok(Self { stft_data, power })
    }

    /// Get the WGSL shader source
    /// Execute the spectrogram operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.stft_data.device();
        let size: usize = self.stft_data.shape().iter().product();
        let num_complex_pairs = size / 2;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.stft_data.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(num_complex_pairs)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            power: f32,
        }

        let params = Params {
            size: num_complex_pairs as u32,
            power: self.power,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Spectrogram Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Spectrogram")
            .shader(SHADER_F64, "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(num_complex_pairs as u32)
            .submit()?;

        // Output shape: [num_complex_pairs] (flattened from original shape)
        let mut output_shape = self.stft_data.shape().to_vec();
        if let Some(last) = output_shape.last_mut() {
            *last /= 2;
        }

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tensor::Tensor;
    #[expect(unused_imports, reason = "conditional imports")]
    use std::sync::Arc;

    #[tokio::test]
    async fn test_spectrogram_basic() {
        // Create complex STFT data: [real, imag, real, imag, ...]
        let device = crate::device::test_pool::get_test_device().await;
        let stft_data = vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0]; // 3 complex pairs, magnitude = 5.0
        let stft_tensor = Tensor::from_vec_on(stft_data, vec![3, 2], device.clone())
            .await
            .unwrap();

        let power_spec = Spectrogram::new(stft_tensor, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(power_spec.shape(), &[3, 1]);
    }

    #[tokio::test]
    async fn test_spectrogram_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single complex pair
        let stft_data = vec![1.0, 0.0];
        let stft_tensor = Tensor::from_vec_on(stft_data, vec![1, 2], device.clone())
            .await
            .unwrap();
        let mag_spec = Spectrogram::new(stft_tensor, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(mag_spec.shape(), &[1, 1]);
    }
}
