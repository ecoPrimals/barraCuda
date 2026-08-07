// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Select - Gather elements along a dimension - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its own dimension and indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Index Select operation - Gather elements along a dimension
///
/// Selects values from input tensor along a dimension using provided indices.
pub struct IndexSelect {
    input: Tensor,
    dim: usize,
    indices: Vec<usize>,
}

impl IndexSelect {
    /// Create a new `IndexSelect` operation
    #[must_use]
    pub fn new(input: Tensor, dim: usize, indices: Vec<usize>) -> Self {
        Self {
            input,
            dim,
            indices,
        }
    }

    /// Execute the index select operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Validate dimension
        if self.dim >= shape.len() {
            let mut expected_shape = shape.to_vec();
            expected_shape.push(self.dim); // Add dim as a hint
            return Err(BarracudaError::invalid_shape(
                expected_shape,
                shape.to_vec(),
            ));
        }

        // Calculate output shape
        let mut output_shape = shape.to_vec();
        output_shape[self.dim] = self.indices.len();

        // Calculate strides
        let dim_size = shape[self.dim];
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let total_size = outer_size * self.indices.len() * inner_size;

        // Convert indices to u32
        let indices_u32: Vec<u32> = self.indices.iter().map(|&i| i as u32).collect();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Indices"),
                contents: bytemuck::cast_slice(&indices_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndexSelect Output"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            total_size: u32,
            dim_size: u32,
            outer_size: u32,
            inner_size: u32,
            num_indices: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            total_size: total_size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
            num_indices: self.indices.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device.create_uniform_buffer("IndexSelect Params", &params);

        ComputeDispatch::new(device, "IndexSelect")
            .shader(include_str!("../shaders/tensor/index_select_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, &input_buffer)
            .storage_read(2, &indices_buffer)
            .storage_rw(3, &output_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, total_size)?;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Select elements along a dimension using indices
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to select from
    /// * `indices` - Indices to select
    ///
    /// # Returns
    ///
    /// Tensor with selected elements
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn index_select_wgsl(self, dim: usize, indices: Vec<usize>) -> Result<Self> {
        IndexSelect::new(self, dim, indices).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_select_1d() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [5] = [0, 1, 2, 3, 4]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![5], device);

        let output = input.index_select_wgsl(0, vec![1, 3, 4]).unwrap();

        assert_eq!(output.shape(), &[3]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }

    #[tokio::test]
    async fn test_index_select_2d_rows() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [3, 2] = [[0,1], [2,3], [4,5]]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![3, 2], device);

        let output = input.index_select_wgsl(0, vec![2, 0]).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        // Row 2: [4, 5], Row 0: [0, 1]
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
    }

    #[tokio::test]
    async fn test_index_select_2d_cols() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input: [2, 3] = [[0,1,2], [3,4,5]]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![2, 3], device);

        let output = input.index_select_wgsl(1, vec![2, 0]).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        // Row 0: [2, 0], Row 1: [5, 3]
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 3.0);
    }
}
