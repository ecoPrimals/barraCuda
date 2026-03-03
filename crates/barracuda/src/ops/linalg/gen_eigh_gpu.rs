//! Generalized Eigenvalue Decomposition (gen_eigh) - GPU-Accelerated Implementation (f64)
//!
//! Solves the generalized eigenvalue problem: Ax = λBx
//! where A is symmetric and B is symmetric positive definite.
//!
//! **Deep Debt Principles**:
//! - ✅ GPU-accelerated eigenvalue decomposition (dominant cost)
//! - ✅ Full f64 precision via SPIR-V/Vulkan
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Hybrid CPU/GPU strategy for optimal performance
//!
//! ## Algorithm
//!
//! Uses Cholesky-based reduction to standard form:
//! ```text
//! Input:  A [N × N] symmetric matrix
//!         B [N × N] symmetric positive definite matrix
//! Output: eigenvalues [N]
//!         eigenvectors [N × N]
//!
//! Steps:
//! 1. Cholesky: B = LLᵀ (CPU - O(n³/3))
//! 2. Transform: C = L⁻¹ A L⁻ᵀ (CPU - O(n³))
//! 3. Eigensolve: Cy = λy (GPU - dominant cost)
//! 4. Back-transform: x = L⁻ᵀ y (CPU - O(n²))
//! ```
//!
//! ## Performance Rationale
//!
//! The eigenvalue decomposition is the bottleneck, requiring O(n³) with a high
//! constant factor due to many Jacobi sweeps. CPU Cholesky and triangular solves
//! are fast enough that GPU overhead isn't worth it for typical sizes.
//!
//! ## Applications
//!
//! - Hartree-Fock-Bogoliubov (HFB) nuclear structure
//! - Vibration analysis in structural mechanics
//! - Quantum chemistry (Roothaan equations)
//!
//! ## References
//!
//! - Golub & Van Loan, "Matrix Computations", Section 8.7
//! - LAPACK DSYGV routine

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::linalg::cholesky::cholesky_f64;
use crate::ops::linalg::BatchedEighGpu;
use std::sync::Arc;

/// Generalized eigenvalue decomposition result (GPU)
#[derive(Debug, Clone)]
pub struct GenEighDecompositionGpu {
    /// Eigenvalues (λ) in ascending order
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (x) as row-major n×n matrix
    /// The i-th eigenvector is column i (accessed as row*n + col)
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension
    pub n: usize,
}

impl GenEighDecompositionGpu {
    /// Get the i-th eigenvector
    pub fn eigenvector(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.n {
            return None;
        }
        let mut v = vec![0.0; self.n];
        for row in 0..self.n {
            v[row] = self.eigenvectors[row * self.n + i];
        }
        Some(v)
    }

    /// Sort eigenvalues and eigenvectors in descending order
    pub fn sort_descending(&mut self) {
        let n = self.n;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.eigenvalues[b]
                .partial_cmp(&self.eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let old_eigenvalues = self.eigenvalues.clone();
        let old_eigenvectors = self.eigenvectors.clone();

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            self.eigenvalues[new_idx] = old_eigenvalues[old_idx];
            for row in 0..n {
                self.eigenvectors[row * n + new_idx] = old_eigenvectors[row * n + old_idx];
            }
        }
    }

    /// Verify that Ax = λBx for all eigenpairs
    pub fn verify(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = self.n;
        let mut max_residual: f64 = 0.0;

        for i in 0..n {
            let lambda = self.eigenvalues[i];
            let x: Vec<f64> = (0..n).map(|row| self.eigenvectors[row * n + i]).collect();

            // Compute Ax
            let mut ax = vec![0.0; n];
            for row in 0..n {
                for col in 0..n {
                    ax[row] += a[row * n + col] * x[col];
                }
            }

            // Compute λBx
            let mut lbx = vec![0.0; n];
            for row in 0..n {
                for col in 0..n {
                    lbx[row] += lambda * b[row * n + col] * x[col];
                }
            }

            let residual_norm: f64 = ax
                .iter()
                .zip(lbx.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let x_norm: f64 = x.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
            let relative_residual = residual_norm / x_norm.max(1e-15);
            max_residual = max_residual.max(relative_residual);
        }

        max_residual
    }
}

/// GPU-accelerated generalized eigenvalue decomposition
///
/// Solves Ax = λBx where A is symmetric and B is symmetric positive definite.
pub struct GenEighGpu;

impl GenEighGpu {
    /// Execute generalized eigenvalue decomposition with GPU-accelerated eigensolve
    ///
    /// # Arguments
    /// * `device` - WgpuDevice to execute on
    /// * `a` - Symmetric matrix A (row-major, n×n) as f64
    /// * `b` - Symmetric positive definite matrix B (row-major, n×n) as f64
    /// * `n` - Matrix dimension
    /// * `max_sweeps` - Maximum Jacobi sweeps for eigensolve (default: 30)
    ///
    /// # Returns
    /// `GenEighDecompositionGpu` with eigenvalues and eigenvectors
    ///
    /// # Errors
    /// - If B is not positive definite (Cholesky fails)
    /// - If matrix dimensions are invalid
    /// - If GPU execution fails
    ///
    /// # Example
    /// ```ignore
    /// use barracuda::ops::linalg::GenEighGpu;
    ///
    /// // Generalized eigenvalue problem: Ax = λBx
    /// let a = vec![4.0, 2.0, 2.0, 3.0]; // [[4, 2], [2, 3]]
    /// let b = vec![2.0, 1.0, 1.0, 2.0]; // [[2, 1], [1, 2]]
    ///
    /// let result = GenEighGpu::execute_f64(device, &a, &b, 2, 30)?;
    /// ```
    pub fn execute_f64(
        device: Arc<WgpuDevice>,
        a: &[f64],
        b: &[f64],
        n: usize,
        max_sweeps: u32,
    ) -> Result<GenEighDecompositionGpu> {
        let expected_len = n * n;
        if a.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Matrix A has wrong size: expected {}, got {}",
                    expected_len,
                    a.len()
                ),
            });
        }
        if b.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Matrix B has wrong size: expected {}, got {}",
                    expected_len,
                    b.len()
                ),
            });
        }

        // Step 1: Cholesky decomposition of B = LLᵀ (GPU)
        let chol = cholesky_f64(device.clone(), b, n)?;
        let l = &chol.l;

        // Step 2: Transform A to standard form: C = L⁻¹ A L⁻ᵀ (CPU)
        // First compute L⁻¹ A by solving L Y = A column by column
        let l_inv_a = solve_lower_triangular_matrix(l, a, n);

        // Then compute C = (L⁻¹ A) L⁻ᵀ
        let c = compute_symmetric_transform(l, &l_inv_a, n);

        // Step 3: GPU eigensolve on transformed matrix C
        // Use batch_size=1 for single matrix
        let (eigenvalues, eigenvectors_c) = BatchedEighGpu::execute_f64(
            device, &c, n, 1, // batch_size = 1
            max_sweeps,
        )?;

        // Step 4: Back-transform eigenvectors: x = L⁻ᵀ y (CPU)
        // Solve Lᵀ x = y for each eigenvector
        let eigenvectors = back_transform_eigenvectors(l, &eigenvectors_c, n);

        Ok(GenEighDecompositionGpu {
            eigenvalues,
            eigenvectors,
            n,
        })
    }

    /// Execute for a batch of generalized eigenvalue problems
    ///
    /// # Arguments
    /// * `device` - WgpuDevice to execute on
    /// * `a_batch` - Packed A matrices [batch_size × n × n]
    /// * `b_batch` - Packed B matrices [batch_size × n × n]
    /// * `n` - Matrix dimension (same for all)
    /// * `batch_size` - Number of matrix pairs
    /// * `max_sweeps` - Maximum Jacobi sweeps
    ///
    /// # Returns
    /// Vector of `GenEighDecompositionGpu`, one per matrix pair
    pub fn execute_batch_f64(
        device: Arc<WgpuDevice>,
        a_batch: &[f64],
        b_batch: &[f64],
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
    ) -> Result<Vec<GenEighDecompositionGpu>> {
        let expected_len = batch_size * n * n;
        if a_batch.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "A batch has wrong size: expected {}, got {}",
                    expected_len,
                    a_batch.len()
                ),
            });
        }
        if b_batch.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "B batch has wrong size: expected {}, got {}",
                    expected_len,
                    b_batch.len()
                ),
            });
        }

        // Transform all matrices to standard form
        let mut c_batch = vec![0.0; expected_len];
        let mut l_batch = vec![0.0; expected_len]; // Store L for back-transform

        for i in 0..batch_size {
            let offset = i * n * n;
            let a = &a_batch[offset..offset + n * n];
            let b = &b_batch[offset..offset + n * n];

            // Cholesky
            let chol = cholesky_f64(device.clone(), b, n)?;
            let l = &chol.l;

            // Store L for back-transform
            l_batch[offset..offset + n * n].copy_from_slice(l);

            // Transform
            let l_inv_a = solve_lower_triangular_matrix(l, a, n);
            let c = compute_symmetric_transform(l, &l_inv_a, n);
            c_batch[offset..offset + n * n].copy_from_slice(&c);
        }

        // GPU eigensolve on all transformed matrices
        let (eigenvalues_batch, eigenvectors_c_batch) =
            BatchedEighGpu::execute_f64(device, &c_batch, n, batch_size, max_sweeps)?;

        // Back-transform all eigenvectors
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let offset = i * n * n;
            let eig_offset = i * n;

            let l = &l_batch[offset..offset + n * n];
            let eigenvectors_c = &eigenvectors_c_batch[offset..offset + n * n];
            let eigenvalues: Vec<f64> = eigenvalues_batch[eig_offset..eig_offset + n].to_vec();

            let eigenvectors = back_transform_eigenvectors(l, eigenvectors_c, n);

            results.push(GenEighDecompositionGpu {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        Ok(results)
    }
}

/// Solve LY = A where L is lower triangular, column by column
fn solve_lower_triangular_matrix(l: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];

    for col in 0..n {
        // Extract column of A
        let mut y = vec![0.0; n];
        for row in 0..n {
            y[row] = a[row * n + col];
        }

        // Forward substitution: Ly = a_col
        for i in 0..n {
            let mut sum = y[i];
            for j in 0..i {
                sum -= l[i * n + j] * y[j];
            }
            let diag = l[i * n + i];
            y[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
        }

        // Store result column
        for row in 0..n {
            result[row * n + col] = y[row];
        }
    }

    result
}

/// Compute C = (L⁻¹ A) L⁻ᵀ
fn compute_symmetric_transform(l: &[f64], l_inv_a: &[f64], n: usize) -> Vec<f64> {
    // First compute L⁻¹
    let mut l_inv = vec![0.0; n * n];
    for i in 0..n {
        let mut x = vec![0.0; n];
        x[i] = 1.0;

        for row in 0..n {
            let mut sum = if row == i { 1.0 } else { 0.0 };
            for k in 0..row {
                sum -= l[row * n + k] * x[k];
            }
            let diag = l[row * n + row];
            x[row] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
        }

        for row in 0..n {
            l_inv[row * n + i] = x[row];
        }
    }

    // C = Y * L⁻ᵀ where Y = L⁻¹ A
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                // Y[i,k] * L⁻ᵀ[k,j] = Y[i,k] * L⁻¹[j,k]
                sum += l_inv_a[i * n + k] * l_inv[j * n + k];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

/// Back-transform eigenvectors: x = L⁻ᵀ y = solve(Lᵀ, y)
fn back_transform_eigenvectors(l: &[f64], eigenvectors_c: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];

    for col in 0..n {
        // Extract eigenvector col
        let mut y = vec![0.0; n];
        for row in 0..n {
            y[row] = eigenvectors_c[row * n + col];
        }

        // Backward substitution: Lᵀ x = y
        // Process from row n-1 down to 0
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                // Lᵀ[i,j] = L[j,i]
                sum -= l[j * n + i] * x[j];
            }
            let diag = l[i * n + i];
            x[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
        }

        // Store result column
        for row in 0..n {
            result[row * n + col] = x[row];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_lower_triangular() {
        // L = [[2, 0], [1, 3]]
        let l = vec![2.0, 0.0, 1.0, 3.0];
        // A = [[1, 0], [0, 1]] (identity)
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let n = 2;

        let result = solve_lower_triangular_matrix(&l, &a, n);

        // L⁻¹ = [[0.5, 0], [-1/6, 1/3]]
        // L⁻¹ * I = L⁻¹
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - (-1.0 / 6.0)).abs() < 1e-10);
        assert!((result[3] - (1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_back_transform() {
        // L = [[2, 0], [0, 2]]
        let l = vec![2.0, 0.0, 0.0, 2.0];
        // eigenvectors = [[1, 0], [0, 1]]
        let eigenvectors = vec![1.0, 0.0, 0.0, 1.0];
        let n = 2;

        let result = back_transform_eigenvectors(&l, &eigenvectors, n);

        // L⁻ᵀ = [[0.5, 0], [0, 0.5]]
        // Result should be scaled by 0.5
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 0.5).abs() < 1e-10);
    }
}
