// SPDX-License-Identifier: AGPL-3.0-or-later
//! Roll - Shift elements along a dimension - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its dimension and shift
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Roll operation - Shift elements along a dimension with wrapping
pub struct Roll {
    input: Tensor,
    shift: i32,
    dim: usize,
}

impl Roll {
    /// Create a new roll operation
    #[must_use]
    pub fn new(input: Tensor, shift: i32, dim: usize) -> Self {
        Self { input, shift, dim }
    }

    /// Execute the roll operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();

        // Calculate dimension parameters
        let dim_size = shape[self.dim];
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            dim_size: u32,
            outer_size: u32,
            inner_size: u32,
            shift: i32,
        }

        let params = Params {
            size: size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
            shift: self.shift,
        };

        let params_buffer = device.create_uniform_buffer("Roll Params", &params);

        ComputeDispatch::new(device, "Roll")
            .shader(include_str!("../shaders/tensor/roll_f64.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Roll elements along a dimension
    /// # Arguments
    /// * `shift` - Number of positions to shift (positive or negative)
    /// * `dim` - Dimension to roll along
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn roll_wgsl(self, shift: i32, dim: usize) -> Result<Self> {
        Roll::new(self, shift, dim).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_roll_1d_positive() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![4], device);

        let output = input.roll_wgsl(1, 0).unwrap();

        assert_eq!(output.shape(), &[4]);
        let result = output.to_vec().unwrap();
        // Roll by 1: [1,2,3,4] -> [4,1,2,3]
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
    }

    #[tokio::test]
    async fn test_roll_1d_negative() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![4], device);

        let output = input.roll_wgsl(-1, 0).unwrap();

        let result = output.to_vec().unwrap();
        // Roll by -1: [1,2,3,4] -> [2,3,4,1]
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 1.0);
    }
}
