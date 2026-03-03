// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cumsum F64 - Cumulative sum along a dimension (double precision)
//!
//! GPU-accelerated prefix sum / cumulative sum for f64 arrays.
//! Useful for numerical integration and cumulative operations.
//!
//! # Algorithm
//!
//! Per-thread sequential scan: each thread handles one (outer, inner) coordinate
//! pair and scans along the specified dimension.
//!
//! # Example
//!
//! ```ignore
//! use barracuda::ops::CumsumF64;
//! use barracuda::prelude::Tensor;
//!
//! // 1D cumsum: [1, 2, 3, 4] → [1, 3, 6, 10]
//! let result = CumsumF64::execute_1d(&device, &[1.0, 2.0, 3.0, 4.0]).await?;
//!
//! // 2D cumsum along axis 0
//! let tensor = Tensor::from_f64_data(&data, vec![3, 4], device.clone())?;
//! let result = CumsumF64::new(tensor, 0).execute().await?;
//! ```

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use wgpu;

/// Parameters passed to the cumsum_f64 shader
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CumsumF64Params {
    size: u32,       // Total number of elements
    dim_size: u32,   // Size of the dimension to scan along
    outer_size: u32, // Product of dimensions before dim
    inner_size: u32, // Product of dimensions after dim
}

/// F64 cumulative sum operation
pub struct CumsumF64 {
    input: Tensor,
    dim: usize,
}

impl CumsumF64 {
    /// Create a new cumsum operation along the specified dimension
    pub fn new(input: Tensor, dim: usize) -> Self {
        Self { input, dim }
    }

    /// WGSL shader source for f64 cumsum
    fn shader() -> &'static str {
        include_str!("../shaders/reduce/cumsum_f64.wgsl")
    }

    /// Execute the cumsum operation
    pub fn execute(self) -> Result<Tensor> {
        let _device = self.input.device();
        let shape = self.input.shape().to_vec();
        let n_dims = shape.len();

        // Validate dimension
        if self.dim >= n_dims {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    self.dim, n_dims
                ),
            });
        }

        let size: usize = shape.iter().product();
        let dim_size = shape[self.dim];

        // Compute outer_size (product of dims before dim) and inner_size (product after)
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();

        // Handle edge cases
        if outer_size == 0 {
            // No outer dimensions - treat as 1
            return self.execute_with_params(size, dim_size, 1, inner_size.max(1));
        }

        self.execute_with_params(size, dim_size, outer_size, inner_size.max(1))
    }

    fn execute_with_params(
        self,
        size: usize,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer
        let output_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("CumsumF64 Output"),
            size: (size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = CumsumF64Params {
            size: size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };

        let params_buffer = device.create_uniform_buffer("CumsumF64 Params", &params);

        let total_pairs = (outer_size * inner_size) as u32;
        let workgroups = total_pairs.div_ceil(WORKGROUP_SIZE_1D);

        ComputeDispatch::new(device.as_ref(), "CumsumF64")
            .shader(Self::shader(), "main")
            .f64()
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit();

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }

    /// Execute 1D cumsum directly on a slice (convenience method)
    pub async fn execute_1d(device: &Arc<WgpuDevice>, data: &[f64]) -> Result<Vec<f64>> {
        let n = data.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let tensor = Tensor::from_f64_data(data, vec![n], device.clone())?;

        let result = Self::new(tensor, 0).execute()?;

        result.to_f64_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_cumsum_f64_1d() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = CumsumF64::execute_1d(&device, &input).await.unwrap();

        // Expected: [1, 3, 6, 10, 15]
        let expected = [1.0, 3.0, 6.0, 10.0, 15.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "Mismatch at {}: got {}, expected {}",
                i,
                got,
                exp
            );
        }
    }

    #[tokio::test]
    async fn test_cumsum_f64_large() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        // Test with larger input
        let n = 10_000;
        let input: Vec<f64> = (1..=n).map(|x| x as f64).collect();
        let result = CumsumF64::execute_1d(&device, &input).await.unwrap();

        // Verify final element: sum(1..=n) = n*(n+1)/2
        let expected_final = (n * (n + 1) / 2) as f64;
        assert!(
            (result[n - 1] - expected_final).abs() < 1e-6,
            "Final cumsum mismatch: got {}, expected {}",
            result[n - 1],
            expected_final
        );

        // Verify a few intermediate values
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[99] - 5050.0).abs() < 1e-6); // sum(1..=100)
    }

    #[tokio::test]
    async fn test_cumsum_f64_precision() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        // Test f64 precision with small increments
        let n = 1000;
        let input: Vec<f64> = vec![1e-10; n];
        let result = CumsumF64::execute_1d(&device, &input).await.unwrap();

        // Final should be n * 1e-10 = 1e-7
        let expected_final = n as f64 * 1e-10;
        let error = (result[n - 1] - expected_final).abs() / expected_final;
        assert!(
            error < 1e-10,
            "Precision error too large: {} (expected {})",
            error,
            expected_final
        );
    }
}
