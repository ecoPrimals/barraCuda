// SPDX-License-Identifier: AGPL-3.0-or-later
//! f64 triangular solve (science-grade precision)

use super::f32::TriangularSolve;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};

/// Triangular solve for f64 data (GPU)
///
/// **Deep Debt Evolution (Feb 16, 2026)**:
/// - Science-grade f64 precision
/// - Native Vulkan fp64 arithmetic
/// - WGSL as unified math language
/// - Includes transpose solve for Cholesky pipeline
pub struct TriangularSolveF64;

impl TriangularSolveF64 {
    /// Solve triangular system L·x = b or U·x = b with f64 precision
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `matrix` - Triangular matrix (row-major f64)
    /// * `rhs` - Right-hand side vector b
    /// * `n` - Matrix/vector dimension
    /// * `lower` - true for lower triangular (forward), false for upper (backward)
    /// * `unit_diagonal` - true if diagonal is implicitly 1.0
    /// # Returns
    /// Solution vector x
    /// # Errors
    /// Returns [`Err`] if `matrix.len() != n * n` or `rhs.len() != n` (invalid dimensions),
    /// if buffer allocation fails, or if GPU dispatch/readback fails (e.g., device lost).
    pub fn execute(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Vec<f64>> {
        if matrix.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * n],
                actual: vec![matrix.len()],
            });
        }
        if rhs.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![rhs.len()],
            });
        }

        // Create buffers
        let matrix_buffer = device.create_buffer_f64(n * n)?;
        device
            .queue
            .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));

        let rhs_buffer = device.create_buffer_f64(n)?;
        device
            .queue
            .write_buffer(&rhs_buffer, 0, bytemuck::cast_slice(rhs));

        let solution_buffer = device.create_buffer_f64(n)?;

        // Params: n, is_lower, is_unit, _pad
        let is_lower = u32::from(lower);
        let is_unit = u32::from(unit_diagonal);
        let params_buffer = device.create_uniform_buffer(
            "TriangularSolve F64 Params",
            &[n as u32, is_lower, is_unit, 0u32],
        );

        ComputeDispatch::new(&device, "TriangularSolve F64")
            .shader(TriangularSolve::wgsl_shader_f64(), "triangular_solve_f64")
            .f64()
            .storage_read(0, &matrix_buffer)
            .storage_read(1, &rhs_buffer)
            .storage_rw(2, &solution_buffer)
            .uniform(3, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        crate::utils::read_buffer_f64(&device, &solution_buffer, n)
    }

    /// Solve L·x = b (forward substitution) with f64
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn forward(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        Self::execute(device, matrix, rhs, n, true, false)
    }

    /// Solve U·x = b (backward substitution) with f64
    /// # Errors
    /// Returns [`Err`] if [`execute`](Self::execute) fails (invalid dimensions, buffer allocation,
    /// or GPU dispatch/readback failure).
    pub fn backward(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        Self::execute(device, matrix, rhs, n, false, false)
    }

    /// Solve Lᵀ·x = b using stored L (transpose solve).
    /// This is the second step of Cholesky solve:
    /// 1. L·z = b (forward)
    /// 2. Lᵀ·x = z (this method)
    ///    The matrix is accessed as transpose internally.
    /// # Errors
    /// Returns [`Err`] if `matrix.len() != n * n` or `rhs.len() != n` (invalid dimensions),
    /// if buffer allocation fails, or if GPU dispatch/readback fails (e.g., device lost).
    pub fn solve_transpose(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        if matrix.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * n],
                actual: vec![matrix.len()],
            });
        }
        if rhs.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![rhs.len()],
            });
        }

        // Create buffers
        let matrix_buffer = device.create_buffer_f64(n * n)?;
        device
            .queue
            .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));

        let rhs_buffer = device.create_buffer_f64(n)?;
        device
            .queue
            .write_buffer(&rhs_buffer, 0, bytemuck::cast_slice(rhs));

        let solution_buffer = device.create_buffer_f64(n)?;

        // Params: n, is_lower=1 (but we use transpose kernel), is_unit=0, _pad
        let params_buffer = device.create_uniform_buffer(
            "TriangularSolve Transpose F64 Params",
            &[n as u32, 1u32, 0u32, 0u32],
        );

        ComputeDispatch::new(&device, "TriangularSolve F64 Transpose")
            .shader(TriangularSolve::wgsl_shader_f64(), "triangular_solve_transpose_f64")
            .f64()
            .storage_read(0, &matrix_buffer)
            .storage_read(1, &rhs_buffer)
            .storage_rw(2, &solution_buffer)
            .uniform(3, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        crate::utils::read_buffer_f64(&device, &solution_buffer, n)
    }

    /// Complete Cholesky solve: Given L from Cholesky(A), solve A·x = b
    /// Performs:
    /// 1. L·z = b (forward substitution)
    /// 2. Lᵀ·x = z (backward with transpose)
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `l_matrix` - Lower triangular Cholesky factor L
    /// * `b` - Right-hand side vector
    /// * `n` - System dimension
    /// # Returns
    /// Solution vector x where A·x = b
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn cholesky_solve(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        l_matrix: &[f64],
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        // Step 1: L·z = b (forward)
        let z = Self::forward(device.clone(), l_matrix, b, n)?;

        // Step 2: Lᵀ·x = z (transpose solve)
        Self::solve_transpose(device, l_matrix, &z, n)
    }
}
