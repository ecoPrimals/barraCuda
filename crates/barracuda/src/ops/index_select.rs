// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Select - Select elements by indices - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Index Select operation - Select elements by indices
pub struct IndexSelect {
    input: Tensor,
    indices: Vec<u32>,
}

impl IndexSelect {
    /// Create a new index select operation
    /// # Errors
    /// Returns [`Err`] if any index is out of bounds for the input size.
    pub fn new(input: Tensor, indices: Vec<u32>) -> Result<Self> {
        let input_size = input.shape().iter().product::<usize>();

        // Validate indices are in bounds
        for &idx in &indices {
            if idx as usize >= input_size {
                return Err(BarracudaError::invalid_op(
                    "IndexSelect",
                    format!("Index {idx} out of bounds for input size {input_size}"),
                ));
            }
        }

        Ok(Self { input, indices })
    }

    /// Execute the index select operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails
    /// (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let num_indices = self.indices.len();

        // Output shape: replace first dimension with number of indices
        let mut output_shape = input_shape.to_vec();
        if output_shape.is_empty() {
            output_shape.push(num_indices);
        } else {
            output_shape[0] = num_indices;
        }

        let output_size = output_shape.iter().product::<usize>();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndexSelect Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters - must match WGSL Params struct (32 bytes)
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

        let input_size = crate::utils::checked_u32(
            input_shape.iter().product::<usize>(),
            "index_select input_size",
        )?;
        let params = Params {
            total_size: crate::utils::checked_u32(output_size, "index_select output_size")?,
            dim_size: input_size,
            outer_size: 1,
            inner_size: 1,
            num_indices: crate::utils::checked_u32(num_indices, "index_select num_indices")?,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "IndexSelect")
            .shader(
                include_str!("../shaders/tensor/index_select_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_read(2, &indices_buffer)
            .storage_rw(3, &output_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Select elements by indices
    /// # Arguments
    /// * `indices` - Indices to select
    /// # Errors
    /// Returns [`Err`] if any index is out of bounds, buffer allocation fails, GPU dispatch fails,
    /// or buffer readback fails (e.g. device lost).
    pub fn index_select(self, indices: Vec<u32>) -> Result<Self> {
        IndexSelect::new(self, indices)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_select_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device);
        let result = input.index_select(vec![0, 2, 4]).unwrap();

        assert_eq!(result.shape(), &[3]);
        let output_data = result.to_vec().unwrap();
        assert_eq!(output_data[0], 1.0);
        assert_eq!(output_data[1], 3.0);
        assert_eq!(output_data[2], 5.0);
    }
}
