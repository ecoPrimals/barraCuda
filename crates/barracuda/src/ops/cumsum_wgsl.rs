// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cumsum - Cumulative sum along a dimension - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its dimension
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Cumsum operation - Cumulative sum along a dimension
pub struct Cumsum {
    input: Tensor,
    dim: usize,
}

impl Cumsum {
    /// Create a new cumsum operation
    #[must_use]
    pub fn new(input: Tensor, dim: usize) -> Self {
        Self { input, dim }
    }

    /// Execute the cumsum operation
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
        }

        let params = Params {
            size: size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };

        let params_buffer = device.create_uniform_buffer("Cumsum Params", &params);

        ComputeDispatch::new(device, "Cumsum")
            .shader(include_str!("../shaders/reduce/cumsum.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d((outer_size * inner_size) as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Compute cumulative sum along a dimension
    /// # Arguments
    /// * `dim` - Dimension to accumulate along
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn cumsum_wgsl(self, dim: usize) -> Result<Self> {
        Cumsum::new(self, dim).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cumsum_1d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![4], device);

        let output = input.cumsum_wgsl(0).unwrap();

        assert_eq!(output.shape(), &[4]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[2], 6.0); // 1 + 2 + 3
        assert_eq!(result[3], 10.0); // 1 + 2 + 3 + 4
    }

    #[tokio::test]
    async fn test_cumsum_2d_dim0() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(data, vec![3, 2], device);

        let output = input.cumsum_wgsl(0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 4.0); // 1 + 3
        assert_eq!(result[3], 6.0); // 2 + 4
        assert_eq!(result[4], 9.0); // 1 + 3 + 5
        assert_eq!(result[5], 12.0); // 2 + 4 + 6
    }
}
