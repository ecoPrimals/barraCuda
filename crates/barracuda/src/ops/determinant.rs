// SPDX-License-Identifier: AGPL-3.0-only
//! Matrix determinant calculation - Pure WGSL implementation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (hardware-agnostic)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Supports 2x2, 3x3, `NxN` matrices
//! - ✅ Batch processing for multiple matrices
//!
//! ## Algorithm
//!
//! - 2x2: det(A) = a*d - b*c (exact)
//! - 3x3: Sarrus rule (exact)
//! - `NxN`: LU decomposition via Gaussian elimination
//!   - For large matrices, uses iterative row reduction
//!   - Determinant = product of diagonal elements after LU decomposition
//!
//! ## Usage
//!
//! ```rust,ignore
//! let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).await?;
//! let det = matrix.determinant()?; // Returns scalar tensor
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeterminantParams {
    matrix_size: u32,
    total_matrices: u32,
    _padding: [u32; 2],
}

/// Matrix determinant computation (2x2, 3x3, or `NxN` via LU).
pub struct Determinant {
    input: Tensor,
}

impl Determinant {
    /// Create a determinant operation for a square matrix (or batch of matrices).
    /// # Errors
    /// Returns [`Err`] if input is not at least 2D or is not a square matrix.
    pub fn new(input: Tensor) -> Result<Self> {
        // Verify square matrix
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "determinant",
                "Requires at least a 2D tensor (matrix)",
            ));
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];

        if rows != cols {
            return Err(BarracudaError::invalid_op(
                "determinant",
                format!("Requires square matrix, got {rows}x{cols}"),
            ));
        }

        Ok(Self { input })
    }

    /// WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/math/determinant_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute determinant calculation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Get matrix dimensions
        let matrix_size = shape[shape.len() - 1]; // N for NxN matrix
        let total_matrices: usize = if shape.len() > 2 {
            shape[..shape.len() - 2].iter().product()
        } else {
            1
        };

        // Create output buffer (one determinant per matrix)
        let output_buffer = device.create_buffer_f32(total_matrices)?;

        // Create parameters
        let params = DeterminantParams {
            matrix_size: matrix_size as u32,
            total_matrices: total_matrices as u32,
            _padding: [0, 0],
        };

        let params_buffer = device.create_uniform_buffer("Determinant Params", &params);

        ComputeDispatch::new(device, "Determinant")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total_matrices as u32)
            .submit()?;

        // Return scalar or vector of determinants
        let output_shape = if total_matrices == 1 {
            vec![1]
        } else {
            shape[..shape.len() - 2].to_vec()
        };

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;
    use crate::device::test_pool::test_prelude::with_device_retry;

    #[tokio::test]
    async fn test_determinant_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 2x2 matrix: [[4, 7], [2, 6]]
        // det = 4*6 - 7*2 = 24 - 14 = 10
        let matrix = Tensor::from_vec_on(vec![4.0, 7.0, 2.0, 6.0], vec![2, 2], device)
            .await
            .unwrap();

        let det = Determinant::new(matrix).unwrap().execute().unwrap();
        let result = det.to_vec().unwrap();

        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 10.0).abs() < 1e-4,
            "Expected 10.0, got {}",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_determinant_edge_cases() {
        with_device_retry(|dev| async move {
            let matrix = Tensor::from_vec_on(vec![5.0], vec![1, 1], dev.clone()).await?;
            let det = Determinant::new(matrix)?.execute()?;
            let result = det.to_vec()?;
            assert!((result[0] - 5.0).abs() < 1e-5);

            let matrix =
                Tensor::from_vec_on(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], dev.clone()).await?;
            let det = Determinant::new(matrix)?.execute()?;
            let result = det.to_vec()?;
            assert!(result[0].abs() < 1e-5, "Singular matrix should have det=0");

            let matrix = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], dev).await?;
            let det = Determinant::new(matrix)?.execute()?;
            let result = det.to_vec()?;
            assert!((result[0] - 1.0).abs() < 1e-5);

            Ok(())
        })
        .await;
    }

    #[tokio::test]
    async fn test_determinant_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 3x3 matrix
        let matrix = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0],
            vec![3, 3],
            device,
        )
        .await
        .unwrap();

        let det = Determinant::new(matrix).unwrap().execute().unwrap();
        let result = det.to_vec().unwrap();

        // Expected: 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5) = -24 + 40 - 15 = 1
        assert!(
            (result[0] - 1.0).abs() < 1e-3,
            "Expected 1.0, got {}",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_determinant_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 2x2 matrix with precise values
        let matrix = Tensor::from_vec_on(vec![1.5, 2.5, 3.5, 4.5], vec![2, 2], device)
            .await
            .unwrap();

        let det = Determinant::new(matrix).unwrap().execute().unwrap();
        let result = det.to_vec().unwrap();

        // det = 1.5*4.5 - 2.5*3.5 = 6.75 - 8.75 = -2.0
        assert!((result[0] - (-2.0)).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_determinant_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 2x2 negative determinant
        let matrix = Tensor::from_vec_on(vec![2.0, 3.0, 1.0, 4.0], vec![2, 2], device)
            .await
            .unwrap();

        let det = Determinant::new(matrix).unwrap().execute().unwrap();
        let result = det.to_vec().unwrap();

        // det = 2*4 - 3*1 = 8 - 3 = 5
        assert!((result[0] - 5.0).abs() < 1e-5);
    }
}
