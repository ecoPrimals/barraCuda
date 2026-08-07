// SPDX-License-Identifier: AGPL-3.0-or-later
//! f32 triangular solve (Tensor-based) and Tensor extension

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Triangular solve operation
///
/// Solves L·x = b (forward) or Uᵀ·x = b (backward)
pub struct TriangularSolve {
    matrix: Tensor, // Triangular matrix (L or U)
    rhs: Tensor,    // Right-hand side vector b
    lower: bool,    // true for lower triangular (forward), false for upper (backward)
}

impl TriangularSolve {
    /// Create new triangular solve operation
    /// # Arguments
    /// * `matrix` - Triangular matrix [N, N]
    /// * `rhs` - Right-hand side vector [N]
    /// * `lower` - true for lower triangular (forward substitution)
    /// # Deep Debt Compliance
    /// - No hardcoded sizes (runtime N)
    /// - No unsafe blocks
    /// - Agnostic design (works with any triangular system)
    #[must_use]
    pub fn new(matrix: Tensor, rhs: Tensor, lower: bool) -> Self {
        Self { matrix, rhs, lower }
    }

    /// Create forward substitution: L·x = b
    #[must_use]
    pub fn forward(matrix: Tensor, rhs: Tensor) -> Self {
        Self::new(matrix, rhs, true)
    }

    /// Create backward substitution: Uᵀ·x = b
    #[must_use]
    pub fn backward(matrix: Tensor, rhs: Tensor) -> Self {
        Self::new(matrix, rhs, false)
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../../shaders/linalg/triangular_solve.wgsl")
    }

    /// Shared with f64 module for transpose solve
    pub(crate) fn wgsl_shader_f64() -> &'static str {
        include_str!("../../../shaders/linalg/triangular_solve_f64.wgsl")
    }

    /// Execute triangular solve on GPU
    /// # Returns
    /// Solution vector x
    /// # Errors
    /// Returns [`Err`] if matrix is not square, rhs size does not match matrix dimension,
    /// buffer allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    /// # Deep Debt Compliance
    /// - Pure WGSL execution (no CPU fallback)
    /// - Capability-based workgroup dispatch
    /// - Safe buffer management
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

        // Create output buffer for solution vector x
        let solution_buffer = device.create_buffer_f32(n)?;

        // Create params buffer with matrix size and substitution type
        let is_lower = u32::from(self.lower);
        let params_buffer =
            device.create_uniform_buffer("TriangularSolve Params", &[n as u32, is_lower]);

        ComputeDispatch::new(device, "TriangularSolve")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.matrix.buffer())
            .storage_read(1, self.rhs.buffer())
            .storage_rw(2, &solution_buffer)
            .uniform(3, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &solution_buffer, n)?;
        Ok(Tensor::new(output_data, vec![n], device.clone()))
    }
}

/// Tensor extension for triangular solve
impl Tensor {
    /// Solve L·x = b (forward substitution)
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    /// # Returns
    /// Solution vector x
    /// # Example
    /// ```ignore
    /// let l = tensor.cholesky()?;  // Get lower triangular L
    /// let x = l.solve_triangular_forward(&b)?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if matrix is not square, rhs size does not match, buffer allocation fails,
    /// GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn solve_triangular_forward(&self, rhs: &Self) -> Result<Self> {
        TriangularSolve::forward(self.clone(), rhs.clone()).execute()
    }

    /// Solve Uᵀ·x = b (backward substitution)
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    /// # Returns
    /// Solution vector x
    /// # Errors
    /// Returns [`Err`] if matrix is not square, rhs size does not match, buffer allocation fails,
    /// GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn solve_triangular_backward(&self, rhs: &Self) -> Result<Self> {
        TriangularSolve::backward(self.clone(), rhs.clone()).execute()
    }

    /// Solve triangular system L·x = b or Uᵀ·x = b
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    /// * `lower` - true for lower triangular, false for upper
    /// # Errors
    /// Returns [`Err`] if matrix is not square, rhs size does not match, buffer allocation fails,
    /// GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn solve_triangular(&self, rhs: &Self, lower: bool) -> Result<Self> {
        TriangularSolve::new(self.clone(), rhs.clone(), lower).execute()
    }
}
