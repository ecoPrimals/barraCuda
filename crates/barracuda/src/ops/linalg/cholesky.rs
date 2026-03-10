// SPDX-License-Identifier: AGPL-3.0-only
//! Cholesky Decomposition - L·Lᵀ factorization - Pure WGSL
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
//! Computes Cholesky decomposition of symmetric positive-definite matrix:
//! ```text
//! Input:  A [N, N] symmetric positive-definite matrix
//! Output: L [N, N] lower triangular matrix such that A = L·Lᵀ
//!
//! Returns zero matrix if input is not positive definite
//! Optimized for scientific computing (N ≤ 30,000)
//! ```
//!
//! ## Precision Support
//!
//! - `execute()` - f32 precision
//! - `execute_f64()` - f64 precision (science-grade, native Vulkan fp64)
//!
//! ## Use Case
//!
//! **RBF Surrogate Learning** (hotSpring physics integration):
//! - Kernel matrix K = L·Lᵀ (step 1 of RBF fit)
//! - Enables efficient solving: K·w = y → L·(Lᵀ·w) = y
//! - GPU-accelerated scientific computing
//!
//! ## References
//!
//! - Golub & Van Loan, "Matrix Computations", Algorithm 4.2.1
//! - Used in scipy.interpolate.RBFInterpolator
//! - hotSpring surrogate learning pipeline

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Cholesky decomposition operation
///
/// Computes L such that A = L·Lᵀ for symmetric positive-definite A
pub struct Cholesky {
    input: Tensor,
}

impl Cholesky {
    /// Create new Cholesky decomposition operation
    /// # Arguments
    /// * `input` - Symmetric positive-definite matrix [N, N]
    /// # Deep Debt Compliance
    /// - No hardcoded sizes (runtime N)
    /// - No unsafe blocks
    /// - Agnostic design (works with any SPD matrix)
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/linalg/cholesky.wgsl")
    }

    fn wgsl_shader_f64() -> &'static str {
        include_str!("../../shaders/linalg/cholesky_f64.wgsl")
    }

    /// Execute Cholesky decomposition on GPU
    /// # Returns
    /// Lower triangular matrix L where A = L·Lᵀ
    /// # Errors
    /// Returns [`Err`] if input is not square (shape not 2D or rows ≠ cols), or if buffer
    /// allocation fails (e.g., device lost, out of memory). Note: returns zero matrix if input
    /// is not positive definite (handled by shader, not an error).
    /// # Deep Debt Compliance
    /// - Pure WGSL execution (no CPU fallback)
    /// - Capability-based workgroup dispatch
    /// - Safe buffer management
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Validate square matrix
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }

        let n = shape[0];
        let size = n * n;

        // Create output buffer for L (lower triangular)
        let output_buffer = device.create_buffer_f32(size)?;

        // Create params buffer with matrix size
        let params_buffer = device.create_uniform_buffer("Cholesky Params", &[n as u32]);

        ComputeDispatch::new(device.as_ref(), "Cholesky")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        // Return lower triangular matrix L
        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }

    /// Execute and also return Lᵀ (useful for solving systems)
    /// # Returns
    /// Tuple (L, Lᵀ) where A = L·Lᵀ
    /// # Errors
    /// Returns [`Err`] if [`execute`](Self::execute) or [`transpose`](crate::tensor::Tensor::transpose)
    /// fails (invalid shape, buffer allocation, or GPU dispatch failure).
    pub fn execute_with_transpose(self) -> Result<(Tensor, Tensor)> {
        let l = self.execute()?;
        let l_t = l.transpose()?;
        Ok((l, l_t))
    }
}

/// Cholesky decomposition for f64 data (GPU)
///
/// **Deep Debt Evolution (Feb 16, 2026)**:
/// - Science-grade f64 precision
/// - Native Vulkan fp64 builtins (sqrt)
/// - WGSL as unified math language
pub struct CholeskyF64;

impl CholeskyF64 {
    /// Execute Cholesky decomposition on GPU with f64 precision
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `data` - Input SPD matrix data (row-major f64)
    /// * `n` - Matrix dimension (n×n)
    /// # Returns
    /// Lower triangular matrix L where A = L·Lᵀ
    /// # Deep Debt Compliance
    /// - Pure WGSL f64 execution
    /// - Native sqrt(f64) on Vulkan
    /// - Hardware-agnostic (NVIDIA/AMD/Intel)
    /// # Errors
    /// Returns [`Err`] if `data.len() != n * n` (invalid dimensions), if buffer allocation fails,
    /// or if GPU dispatch/readback fails (e.g., device lost).
    pub fn execute(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        data: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        if data.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * n],
                actual: vec![data.len()],
            });
        }

        // Create input buffer with f64 data
        let input_buffer = device.create_buffer_f64(n * n)?;
        device
            .queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(data));

        // Create output buffer for L
        let output_buffer = device.create_buffer_f64(n * n)?;

        // Create params buffer
        let params_buffer = device.create_uniform_buffer("Cholesky F64 Params", &[n as u32]);

        ComputeDispatch::new(device.as_ref(), "Cholesky F64")
            .shader(Cholesky::wgsl_shader_f64(), "cholesky_f64")
            .f64()
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        // Read back f64 results
        crate::utils::read_buffer_f64(&device, &output_buffer, n * n)
    }

    /// Execute batched Cholesky decomposition
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `data` - Batch of SPD matrices (flattened: `batch_size` × n × n)
    /// * `n` - Matrix dimension per batch element
    /// * `batch_size` - Number of matrices
    /// # Returns
    /// Batch of lower triangular matrices L
    /// # Errors
    /// Returns [`Err`] if `data.len() != batch_size * n * n` (invalid dimensions), if buffer
    /// allocation fails, or if GPU dispatch/readback fails (e.g., device lost).
    pub fn execute_batch(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        data: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        let mat_size = n * n;
        if data.len() != batch_size * mat_size {
            return Err(BarracudaError::InvalidShape {
                expected: vec![batch_size * mat_size],
                actual: vec![data.len()],
            });
        }

        // Create buffers
        let input_buffer = device.create_buffer_f64(batch_size * mat_size)?;
        device
            .queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(data));

        let output_buffer = device.create_buffer_f64(batch_size * mat_size)?;

        let params_buffer = device.create_uniform_buffer("Cholesky F64 Batch Params", &[n as u32]);

        ComputeDispatch::new(device.as_ref(), "Cholesky F64 Batch")
            .shader(Cholesky::wgsl_shader_f64(), "cholesky_f64_batched")
            .f64()
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(batch_size as u32, 1, 1)
            .submit()?;

        crate::utils::read_buffer_f64(&device, &output_buffer, batch_size * mat_size)
    }
}

/// Tensor extension for Cholesky decomposition
impl Tensor {
    /// Compute Cholesky decomposition: A = L·Lᵀ
    /// # Returns
    /// Lower triangular matrix L
    /// # Errors
    /// Returns [`Err`] if [`Cholesky::execute`] fails (non-square input, buffer allocation,
    /// or GPU dispatch failure).
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], vec![2, 2], device)?;
    /// let l = a.cholesky()?;
    /// // l ≈ [[2.0, 0.0], [1.0, 1.414]]
    /// ```
    pub fn cholesky(self) -> Result<Self> {
        Cholesky::new(self).execute()
    }

    /// Compute Cholesky decomposition with transpose
    /// # Returns
    /// Tuple (L, Lᵀ)
    /// # Errors
    /// Returns [`Err`] if [`Cholesky::execute_with_transpose`] fails (non-square input, buffer
    /// allocation, or GPU dispatch failure).
    pub fn cholesky_with_transpose(self) -> Result<(Self, Self)> {
        Cholesky::new(self).execute_with_transpose()
    }
}

#[cfg(test)]
#[path = "cholesky_tests.rs"]
mod tests;
