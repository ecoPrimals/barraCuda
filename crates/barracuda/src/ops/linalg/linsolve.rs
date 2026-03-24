// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear System Solve - Gaussian elimination with partial pivoting
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Runtime-configured matrix size
//! - ✅ Capability-based dispatch
//!
//! ## Algorithm
//!
//! Solves linear system A·x = b:
//! ```text
//! Input 1: matrix A [N, N]
//! Input 2: vector b [N]
//! Output: solution x [N] where A·x = b
//!
//! Uses Gaussian elimination with partial pivoting
//! (LU decomposition with row pivoting, forward/backward substitution)
//! ```
//!
//! ## Use Case
//!
//! **Scientific Computing**:
//! - General linear system solving
//! - Alternative to Cholesky solve for non-SPD matrices
//!
//! ## References
//!
//! - Golub & Van Loan, "Matrix Computations", Section 3.2
//! - scipy.linalg.solve

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use crate::utils;

/// Linear system solve operation
///
/// Solves A·x = b for square matrix A and vector b
pub struct LinSolve {
    matrix: Tensor,
    rhs: Tensor,
}

impl LinSolve {
    /// Create new linear solve operation
    /// # Arguments
    /// * `matrix` - Square matrix A [N, N]
    /// * `rhs` - Right-hand side vector b [N]
    /// # Deep Debt Compliance
    /// - No hardcoded sizes (runtime N)
    /// - No unsafe blocks
    /// - Agnostic design (works with any square system)
    #[must_use]
    pub fn new(matrix: Tensor, rhs: Tensor) -> Self {
        Self { matrix, rhs }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/linalg/linsolve.wgsl")
    }

    /// Execute linear solve on GPU
    /// # Returns
    /// Solution vector x where A·x = b
    /// # Errors
    /// - Returns error if matrix is not square
    /// - Returns error if rhs length doesn't match matrix dimension
    /// - Returns zero vector if matrix is singular
    /// # Deep Debt Compliance
    /// - Pure WGSL execution (no CPU fallback)
    /// - Capability-based workgroup dispatch
    /// - Safe buffer management
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.matrix.device();
        let matrix_shape = self.matrix.shape();
        let rhs_shape = self.rhs.shape();

        // Validate square matrix
        if matrix_shape.len() != 2 || matrix_shape[0] != matrix_shape[1] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: matrix_shape.to_vec(),
            });
        }

        let n = matrix_shape[0];

        // Validate rhs is a vector of length n
        if rhs_shape.len() != 1 || rhs_shape[0] != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: rhs_shape.to_vec(),
            });
        }

        // Output buffer: work matrix (n*n) + solution (n) = n*n + n
        let output_size = n * n + n;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params buffer
        let params_buffer = device.create_uniform_buffer("LinSolve Params", &[n as u32]);

        ComputeDispatch::new(device.as_ref(), "LinSolve")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.matrix.buffer())
            .storage_read(1, self.rhs.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        // Read solution from output buffer (last n elements)
        let full_data = utils::read_buffer(device, &output_buffer, output_size)?;
        let solution: Vec<f32> = full_data[n * n..].to_vec();

        Ok(Tensor::new(solution, vec![n], device.clone()))
    }
}

/// Tensor extension for linear solve
impl Tensor {
    /// Solve linear system A·x = b
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    /// # Returns
    /// Solution vector x
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_vec(vec![2.0, 1.0, 1.0, 2.0], vec![2, 2], device)?;
    /// let b = Tensor::from_vec(vec![5.0, 4.0], vec![2], device)?;
    /// let x = a.linsolve(&b)?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn linsolve(&self, rhs: &Self) -> Result<Self> {
        LinSolve::new(self.clone(), rhs.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    const LINSOLVE_SHADER: &str = include_str!("../../shaders/linalg/linsolve.wgsl");

    #[test]
    fn linsolve_shader_source_valid() {
        assert!(!LINSOLVE_SHADER.is_empty());
        assert!(LINSOLVE_SHADER.contains("fn ") || LINSOLVE_SHADER.contains("@compute"));
    }

    #[tokio::test]
    async fn test_linsolve_identity() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // I·x = b => x = b
        let i_data = vec![1.0, 0.0, 0.0, 1.0];
        let b_data = vec![3.0, 7.0];

        let matrix = Tensor::from_vec_on(i_data, vec![2, 2], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(b_data.clone(), vec![2], device)
            .await
            .unwrap();

        let x = matrix.linsolve(&rhs).unwrap();
        let solution = x.to_vec().unwrap();

        assert_eq!(solution.len(), 2);
        assert!((solution[0] - 3.0).abs() < 1e-5);
        assert!((solution[1] - 7.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_linsolve_2x2() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // [[2, 1], [1, 2]] · x = [5, 4]
        // Solution: x = [2, 1]
        // Verification: 2*2+1*1=5, 1*2+2*1=4 ✓
        let a_data = vec![2.0, 1.0, 1.0, 2.0];
        let b_data = vec![5.0, 4.0];

        let matrix = Tensor::from_vec_on(a_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(b_data.clone(), vec![2], device.clone())
            .await
            .unwrap();

        let x = matrix.linsolve(&rhs).unwrap();
        let solution = x.to_vec().unwrap();

        assert_eq!(solution.len(), 2);
        assert!(
            (solution[0] - 2.0).abs() < 1e-5,
            "x[0] should be 2.0, got {}",
            solution[0]
        );
        assert!(
            (solution[1] - 1.0).abs() < 1e-5,
            "x[1] should be 1.0, got {}",
            solution[1]
        );

        // Verify A·x ≈ b
        let x_2d = x.unsqueeze(1).unwrap();
        let ax = matrix.matmul(&x_2d).unwrap().squeeze().unwrap();
        let ax_data = ax.to_vec().unwrap();
        for (i, (&exp, &act)) in b_data.iter().zip(ax_data.iter()).enumerate() {
            assert!(
                (exp - act).abs() < 1e-4,
                "A·x[{i}] should be {exp}, got {act}"
            );
        }
    }

    #[tokio::test]
    async fn test_linsolve_3x3() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Simple 3x3: [[1,0,0],[0,1,0],[0,0,1]]·x = [1,2,3] => x = [1,2,3]
        let a_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b_data = vec![1.0, 2.0, 3.0];

        let matrix = Tensor::from_vec_on(a_data, vec![3, 3], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(b_data.clone(), vec![3], device)
            .await
            .unwrap();

        let x = matrix.linsolve(&rhs).unwrap();
        let solution = x.to_vec().unwrap();

        assert_eq!(solution.len(), 3);
        assert!((solution[0] - 1.0).abs() < 1e-5);
        assert!((solution[1] - 2.0).abs() < 1e-5);
        assert!((solution[2] - 3.0).abs() < 1e-5);
    }
}
