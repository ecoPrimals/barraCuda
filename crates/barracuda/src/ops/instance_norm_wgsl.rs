// SPDX-License-Identifier: AGPL-3.0-or-later
//! Instance Normalization - Normalize per instance - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its normalization parameters
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Instance normalization operation
pub struct InstanceNorm {
    input: Tensor,
    epsilon: f32,
}

impl InstanceNorm {
    /// Create a new instance normalization operation
    #[must_use]
    pub fn new(input: Tensor, epsilon: f32) -> Self {
        Self { input, epsilon }
    }

    /// Execute the instance normalization operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();

        // Assume NCHW format: [batch, channels, height, width]
        let batch = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch: u32,
            channels: u32,
            spatial_size: u32,
            epsilon: f32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            spatial_size: spatial_size as u32,
            epsilon: self.epsilon,
        };

        let params_buffer = device.create_uniform_buffer("InstanceNorm Params", &params);

        ComputeDispatch::new(device, "InstanceNorm")
            .shader(include_str!("../shaders/norm/instance_norm_f64.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d((batch * channels) as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Apply instance normalization (normalize per instance in NCHW format)
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Small constant for numerical stability (default: 1e-5)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn instance_norm_wgsl(self, epsilon: f32) -> Result<Self> {
        InstanceNorm::new(self, epsilon).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_instance_norm_simple() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1 batch, 1 channel, 2x2 spatial
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 2, 2], device);

        let output = input.instance_norm_wgsl(1e-5).unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);

        // Check that mean is ~0
        let result = output.to_vec().unwrap();
        let mean: f32 = result.iter().sum::<f32>() / 4.0;
        assert!((mean).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_instance_norm_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        // 2 batches, 1 channel each, 2x1 spatial
        let data = vec![
            1.0, 2.0, // batch 0, channel 0
            3.0, 4.0, // batch 1, channel 0
        ];
        let input = Tensor::new(data, vec![2, 1, 2, 1], device);

        let output = input.instance_norm_wgsl(1e-5).unwrap();

        assert_eq!(output.shape(), &[2, 1, 2, 1]);

        // Each instance should be normalized independently
        let result = output.to_vec().unwrap();

        // First instance mean should be ~0
        let mean1 = f32::midpoint(result[0], result[1]);
        assert!((mean1).abs() < 1e-5);

        // Second instance mean should be ~0
        let mean2 = f32::midpoint(result[2], result[3]);
        assert!((mean2).abs() < 1e-5);
    }
}
