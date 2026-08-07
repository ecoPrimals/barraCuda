// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gather - Select elements using indices - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its dimension and indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Gather operation - Select elements using indices
pub struct Gather {
    input: Tensor,
    dim: usize,
    indices: Vec<u32>,
}

impl Gather {
    /// Create a new gather operation
    #[must_use]
    pub fn new(input: Tensor, dim: usize, indices: Vec<u32>) -> Self {
        Self {
            input,
            dim,
            indices,
        }
    }

    /// Execute the gather operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Calculate dimension parameters
        let dim_size = shape[self.dim];
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let gather_size = self.indices.len();

        let output_size = outer_size * gather_size * inner_size;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gather Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gather Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            dim_size: u32,
            outer_size: u32,
            inner_size: u32,
            gather_size: u32,
        }

        let params = Params {
            size: output_size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
            gather_size: gather_size as u32,
        };

        let params_buffer = device.create_uniform_buffer("Gather Params", &params);

        ComputeDispatch::new(device, "Gather")
            .shader(include_str!("../shaders/tensor/gather_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_read(1, &indices_buffer)
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        // Calculate output shape
        let mut output_shape = shape.to_vec();
        output_shape[self.dim] = gather_size;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Gather elements along a dimension using indices
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to gather from
    /// * `indices` - Indices to gather
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn gather_wgsl(self, dim: usize, indices: Vec<u32>) -> Result<Self> {
        Gather::new(self, dim, indices).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gather_1d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![5], device);

        let output = input.gather_wgsl(0, vec![0, 2, 4]).unwrap();

        assert_eq!(output.shape(), &[3]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 5.0);
    }

    #[tokio::test]
    async fn test_gather_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(data, vec![3, 2], device);

        let output = input.gather_wgsl(0, vec![0, 2]).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        // Original: [[1,2], [3,4], [5,6]]
        // Gather indices [0, 2]: [[1,2], [5,6]]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 6.0);
    }
}
