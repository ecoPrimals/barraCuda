// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repeat - Repeat tensor along dimensions - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its repeat counts
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Repeat operation - Repeat tensor along dimensions
pub struct Repeat {
    input: Tensor,
    repeats: Vec<usize>,
}

impl Repeat {
    /// Create a new repeat operation
    #[must_use]
    pub fn new(input: Tensor, repeats: Vec<usize>) -> Self {
        Self { input, repeats }
    }

    /// Execute the repeat operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_size: usize = shape.iter().product();

        // Calculate output shape and size
        let mut output_shape = shape.to_vec();
        for (i, &repeat) in self.repeats.iter().enumerate() {
            if i < output_shape.len() {
                output_shape[i] *= repeat;
            }
        }
        let output_size: usize = output_shape.iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Repeat Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters (support up to 4D)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_size: u32,
            output_size: u32,
            num_dims: u32,
            _pad: u32,
            dim_sizes: [u32; 4],
            repeats: [u32; 4],
        }

        let mut dim_sizes = [1u32; 4];
        let mut repeats = [1u32; 4];

        for (i, &size) in shape.iter().enumerate().take(4) {
            dim_sizes[i] = size as u32;
        }
        for (i, &repeat) in self.repeats.iter().enumerate().take(4) {
            repeats[i] = repeat as u32;
        }

        let params = Params {
            input_size: input_size as u32,
            output_size: output_size as u32,
            num_dims: shape.len().min(4) as u32,
            _pad: 0,
            dim_sizes,
            repeats,
        };

        let params_buffer = device.create_uniform_buffer("Repeat Params", &params);

        ComputeDispatch::new(device, "Repeat")
            .shader(include_str!("../shaders/tensor/repeat_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Repeat tensor along dimensions
    /// # Arguments
    /// * `repeats` - Number of times to repeat each dimension
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn repeat_wgsl(self, repeats: Vec<usize>) -> Result<Self> {
        Repeat::new(self, repeats).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repeat_1d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0];
        let input = Tensor::from_data(&data, vec![3], device).unwrap();

        let output = input.repeat_wgsl(vec![2]).unwrap();

        assert_eq!(output.shape(), &[6]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result[4], 3.0);
        assert_eq!(result[5], 3.0);
    }

    #[tokio::test]
    async fn test_repeat_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&data, vec![2, 2], device).unwrap();

        let output = input.repeat_wgsl(vec![2, 1]).unwrap();

        assert_eq!(output.shape(), &[4, 2]);
        let result = output.to_vec().unwrap();
        // Original: [[1,2], [3,4]]
        // Repeated on dim 0: [[1,2], [1,2], [3,4], [3,4]]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
    }
}
