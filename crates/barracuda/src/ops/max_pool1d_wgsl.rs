// SPDX-License-Identifier: AGPL-3.0-or-later
//! Max Pool 1D - Temporal max pooling - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its own requirements (kernel size, stride)
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Max Pool 1D operation - Temporal max pooling
///
/// Applies 1D max pooling over an input signal (batch, channels, length).
pub struct MaxPool1D {
    input: Tensor,
    kernel_size: usize,
    stride: usize,
}

impl MaxPool1D {
    /// Create a new `MaxPool1D` operation
    #[must_use]
    pub fn new(input: Tensor, kernel_size: usize, stride: usize) -> Self {
        Self {
            input,
            kernel_size,
            stride,
        }
    }

    /// Execute the max pool 1D operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Validate input shape (batch, channels, length)
        if shape.len() != 3 {
            return Err(BarracudaError::invalid_shape(
                vec![0, 0, 0], // Expected: 3D tensor
                shape.to_vec(),
            ));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let input_size = shape[2];

        // Calculate output size
        let output_size = (input_size - self.kernel_size) / self.stride + 1;
        let total_elements = batch_size * channels * output_size;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxPool1D Output"),
            size: (total_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_size: u32,
            output_size: u32,
            channels: u32,
            batch_size: u32,
            kernel_size: u32,
            stride: u32,
        }

        let params = Params {
            input_size: input_size as u32,
            output_size: output_size as u32,
            channels: channels as u32,
            batch_size: batch_size as u32,
            kernel_size: self.kernel_size as u32,
            stride: self.stride as u32,
        };

        let params_buffer = device.create_uniform_buffer("MaxPool1D Params", &params);

        ComputeDispatch::new(device, "MaxPool1D")
            .shader(include_str!("../shaders/pooling/max_pool1d_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total_elements as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, total_elements)?;

        Ok(Tensor::new(
            output_data,
            vec![batch_size, channels, output_size],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 1D max pooling over the tensor
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Size of the pooling window
    /// * `stride` - Stride of the pooling operation
    ///
    /// # Returns
    ///
    /// Pooled tensor with shape (batch, channels, `output_length`)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn max_pool1d_wgsl(self, kernel_size: usize, stride: usize) -> Result<Self> {
        MaxPool1D::new(self, kernel_size, stride).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_max_pool1d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [1, 1, 8] - single batch, single channel, 8 elements
        let data = vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 6.0, 2.0];
        let input = Tensor::new(data, vec![1, 1, 8], device);

        let output = input.max_pool1d_wgsl(2, 2).unwrap();

        assert_eq!(output.shape(), &[1, 1, 4]);
        let result = output.to_vec().unwrap();
        // Max of [1,3]=3, [2,4]=4, [5,1]=5, [6,2]=6
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 6.0);
    }

    #[tokio::test]
    async fn test_max_pool1d_multi_channel() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [1, 2, 4] - single batch, 2 channels, 4 elements each
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // Channel 0
            4.0, 3.0, 2.0, 1.0, // Channel 1
        ];
        let input = Tensor::new(data, vec![1, 2, 4], device);

        let output = input.max_pool1d_wgsl(2, 2).unwrap();

        assert_eq!(output.shape(), &[1, 2, 2]);
        let result = output.to_vec().unwrap();
        // Channel 0: max([1,2])=2, max([3,4])=4
        // Channel 1: max([4,3])=4, max([2,1])=2
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 2.0);
    }

    #[tokio::test]
    async fn test_max_pool1d_stride_one() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [1, 1, 5] - overlapping windows
        let data = vec![1.0, 5.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 5], device);

        let output = input.max_pool1d_wgsl(3, 1).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3]);
        let result = output.to_vec().unwrap();
        // max([1,5,2])=5, max([5,2,3])=5, max([2,3,4])=4
        assert_eq!(result[0], 5.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 4.0);
    }
}
