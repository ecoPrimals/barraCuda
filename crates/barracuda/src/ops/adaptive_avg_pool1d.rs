// SPDX-License-Identifier: AGPL-3.0-or-later
//! `AdaptiveAvgPool1D` - 1D Adaptive Average Pooling
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Applies average pooling with adaptive kernel size to produce fixed output size
//! Used in models like `ResNet`, VGG for variable input sizes

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaptiveAvgPool1DParams {
    batch_size: u32,
    channels: u32,
    in_length: u32,
    out_length: u32,
}

/// 1D adaptive average pooling — fixed output length regardless of input.
pub struct AdaptiveAvgPool1D {
    input: Tensor,
    output_length: usize,
}

impl AdaptiveAvgPool1D {
    /// Create adaptive avg pool. Input must be 3D [B, C, L].
    /// # Errors
    /// Returns [`Err`] if input is not 3D [B, C, L] or `output_length` is zero.
    pub fn new(input: Tensor, output_length: usize) -> Result<Self> {
        // Validate input shape: must be 3D [B, C, L]
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(BarracudaError::invalid_op(
                "adaptive_avg_pool1d",
                "input must be 3D tensor [B, C, L]",
            ));
        }

        if output_length == 0 {
            return Err(BarracudaError::invalid_op(
                "adaptive_avg_pool1d",
                "output_length must be positive",
            ));
        }

        Ok(Self {
            input,
            output_length,
        })
    }

    /// Execute adaptive average pooling on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_length = shape[2];

        let output_size = batch_size * channels * self.output_length;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = AdaptiveAvgPool1DParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_length: in_length as u32,
            out_length: self.output_length as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adaptive_avg_pool1d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "AdaptiveAvgPool1D")
            .shader(
                include_str!("../shaders/pooling/adaptive_avg_pool1d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, self.output_length],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 1D adaptive average pooling to fixed output length
    /// # Errors
    /// Returns [`Err`] if input is not 3D [B, C, L], `output_length` is zero, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn adaptive_avg_pool1d(self, output_length: usize) -> Result<Self> {
        AdaptiveAvgPool1D::new(self, output_length)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adaptive_avg_pool1d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![1.0; 3 * 16];
        let input = Tensor::from_vec_on(input_data, vec![1, 3, 16], device.clone())
            .await
            .unwrap();

        let output = input.adaptive_avg_pool1d(8).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 8]);
        assert_eq!(result.len(), 24);
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_adaptive_avg_pool1d_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Invalid shape (not 3D)
        let input = Tensor::from_vec_on(vec![1.0; 16], vec![4, 4], device.clone())
            .await
            .unwrap();
        assert!(input.adaptive_avg_pool1d(8).is_err());
    }
}
