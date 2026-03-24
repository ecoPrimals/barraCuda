// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generalized Symmetric Eigenvalue Problem (f64 CPU)
//!
//! Solves the generalized eigenvalue problem: Ax = λBx
//! where A is symmetric and B is symmetric positive definite.
//!
//! # Algorithm
//!
//! Uses Cholesky-based reduction to standard form:
//!
//! 1. Compute Cholesky decomposition: B = `LLᵀ`
//! 2. Transform: C = L⁻¹ A (L⁻¹)ᵀ (symmetric)
//! 3. Solve standard eigenproblem: Cy = λy
//! 4. Back-transform eigenvectors: x = (L⁻¹)ᵀ y
//!
//! # Applications
//!
//! - Hartree-Fock-Bogoliubov (HFB) nuclear structure
//! - Vibration analysis in structural mechanics
//! - Principal component analysis with constraints
//! - Quantum chemistry (Roothaan equations)
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations", Section 8.7
//! - Numerical Recipes, 3rd Edition, Section 11.0
//! - LAPACK DSYGV/DSYGVX routines

use crate::error::{BarracudaError, Result};
use crate::linalg::cholesky::{CholeskyDecomposition, cholesky_f64};
use crate::linalg::eigh::eigh_f64;
use std::sync::Arc;

use crate::device::WgpuDevice;

/// Generalized eigenvalue decomposition result
#[derive(Debug, Clone)]
pub struct GenEighDecomposition {
    /// Eigenvalues (λ) in ascending order
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (x) as column-major n×n matrix
    /// The i-th eigenvector is column i
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension
    pub n: usize,
}

impl GenEighDecomposition {
    /// Get the i-th eigenvector
    /// Eigenvectors are stored as columns in row-major format:
    /// eigenvector j has component i at position [i * n + j]
    #[must_use]
    pub fn eigenvector(&self, i: usize) -> Option<Vec<f64>> {
        if i >= self.n {
            return None;
        }
        let mut v = vec![0.0; self.n];
        for (row, v_row) in v.iter_mut().enumerate() {
            *v_row = self.eigenvectors[row * self.n + i];
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
            // Copy column old_idx to column new_idx
            for row in 0..n {
                self.eigenvectors[row * n + new_idx] = old_eigenvectors[row * n + old_idx];
            }
        }
    }

    /// Verify that Ax = λBx for all eigenpairs
    /// Returns the maximum residual ||Ax - λBx|| / ||x||
    #[must_use]
    pub fn verify(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = self.n;
        let mut max_residual: f64 = 0.0;

        for i in 0..n {
            let lambda = self.eigenvalues[i];

            // Extract eigenvector i (column i in row-major format)
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

            // Compute ||Ax - λBx||
            let residual_norm: f64 = ax
                .iter()
                .zip(lbx.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            // Compute ||x||
            let x_norm: f64 = x.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();

            let relative_residual = residual_norm / x_norm.max(1e-15);
            max_residual = max_residual.max(relative_residual);
        }

        max_residual
    }

    /// Compute the trace of the generalized eigenvalue problem
    /// (sum of eigenvalues)
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }
}

/// Solve the generalized symmetric eigenvalue problem Ax = λBx
///
/// # Arguments
///
/// * `a` - Symmetric matrix A (row-major, n×n)
/// * `b` - Symmetric positive definite matrix B (row-major, n×n)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// `GenEighDecomposition` with eigenvalues and eigenvectors
///
/// # Errors
///
/// - If B is not positive definite (Cholesky fails)
/// - If the transformed matrix has issues (eigh fails)
/// - If dimensions are invalid
///
/// # Example
///
/// ```no_run
/// use barracuda::linalg::gen_eigh_f64;
/// use barracuda::prelude::WgpuDevice;
/// use std::sync::Arc;
///
/// # async fn example() -> barracuda::error::Result<()> {
/// let device = Arc::new(WgpuDevice::new().await?);
/// let a = vec![2.0, 1.0, 1.0, 2.0];
/// let b = vec![1.0, 0.0, 0.0, 1.0];
///
/// let result = gen_eigh_f64(device, &a, &b, 2)?;
///
/// assert!((result.eigenvalues[0] - 1.0).abs() < 1e-10);
/// assert!((result.eigenvalues[1] - 3.0).abs() < 1e-10);
/// # Ok(())
/// # }
/// ```
pub fn gen_eigh_f64(
    device: Arc<WgpuDevice>,
    a: &[f64],
    b: &[f64],
    n: usize,
) -> Result<GenEighDecomposition> {
    if a.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Matrix A has wrong size: expected {}, got {}",
                n * n,
                a.len()
            ),
        });
    }
    if b.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Matrix B has wrong size: expected {}, got {}",
                n * n,
                b.len()
            ),
        });
    }

    // Step 1: Cholesky decomposition of B = LLᵀ (GPU)
    let chol = cholesky_f64(device, b, n)?;

    // Step 2: Compute C = L⁻¹ A (L⁻¹)ᵀ
    // First compute L⁻¹ A by solving L Y = A for each column of A
    let l_inv_a = solve_lower_triangular_matrix(&chol, a, n)?;

    // Then compute (L⁻¹ A) (L⁻¹)ᵀ = L⁻¹ A L⁻ᵀ
    // This is equivalent to solving L Z = (L⁻¹ A)ᵀ, then transposing
    let c = compute_symmetric_transform(&chol, &l_inv_a, n)?;

    // Step 3: Solve standard eigenproblem Cy = λy
    let std_eigh = eigh_f64(&c, n)?;

    // Step 4: Back-transform eigenvectors x = (L⁻¹)ᵀ y = L⁻ᵀ y
    // This is equivalent to solving Lᵀ x = y for each eigenvector
    let eigenvectors = back_transform_eigenvectors(&chol, &std_eigh.eigenvectors, n)?;

    Ok(GenEighDecomposition {
        eigenvalues: std_eigh.eigenvalues,
        eigenvectors,
        n,
    })
}

/// Solve LY = A where L is lower triangular (from Cholesky), column by column
fn solve_lower_triangular_matrix(
    chol: &CholeskyDecomposition,
    a: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let l = &chol.l;
    let mut result = vec![0.0; n * n];

    // Solve for each column of A
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
            y[i] = sum / l[i * n + i];
        }

        // Store result column
        for row in 0..n {
            result[row * n + col] = y[row];
        }
    }

    Ok(result)
}

/// Compute C = (L⁻¹ A) (L⁻¹)ᵀ = Y Yᵀ where Y = L⁻¹ A
/// But we want C = L⁻¹ A L⁻ᵀ, so we need to solve Lᵀ Z = Yᵀ row by row
fn compute_symmetric_transform(
    chol: &CholeskyDecomposition,
    l_inv_a: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let l = &chol.l;
    let mut c = vec![0.0; n * n];

    // For each row of C, solve Lᵀ z = (L⁻¹ A)[row,:]ᵀ
    // Then C[row,:] = zᵀ
    // But since C should be symmetric, we can compute C = (L⁻¹ A) (L⁻ᵀ)
    // which is (L⁻¹ A) (L⁻¹)ᵀ

    // Actually, let's compute it directly:
    // C[i,j] = sum_k (L⁻¹ A)[i,k] * (L⁻ᵀ)[k,j]
    //        = sum_k Y[i,k] * (L⁻¹)[j,k]

    // Compute L⁻¹ (inverse of lower triangular)
    let mut l_inv = vec![0.0; n * n];
    for i in 0..n {
        // Solve L x = e_i
        let mut x = vec![0.0; n];
        x[i] = 1.0;

        for row in 0..n {
            let mut sum = if row == i { 1.0 } else { 0.0 };
            for k in 0..row {
                sum -= l[row * n + k] * x[k];
            }
            x[row] = sum / l[row * n + row];
        }

        // Store as column i of L⁻¹
        for row in 0..n {
            l_inv[row * n + i] = x[row];
        }
    }

    // C = Y * L⁻ᵀ = (L⁻¹ A) * (L⁻¹)ᵀ
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

    // Ensure symmetry (average out numerical errors)
    for i in 0..n {
        for j in i + 1..n {
            let avg = 0.5 * (c[i * n + j] + c[j * n + i]);
            c[i * n + j] = avg;
            c[j * n + i] = avg;
        }
    }

    Ok(c)
}

/// Back-transform eigenvectors: x = L⁻ᵀ y (solve Lᵀ x = y)
///
/// Note: eigenvectors from `eigh_f64` are stored as columns in row-major format:
/// eigenvector j has component i at position [i * n + j]
fn back_transform_eigenvectors(
    chol: &CholeskyDecomposition,
    y: &[f64],
    n: usize,
) -> Result<Vec<f64>> {
    let l = &chol.l;
    let mut x = vec![0.0; n * n];

    // For each eigenvector (column j)
    for j in 0..n {
        // Extract eigenvector j: y_j[i] = y[i * n + j]
        let mut xi = vec![0.0; n];
        for i in 0..n {
            xi[i] = y[i * n + j];
        }

        // Solve Lᵀ x = y (backward substitution)
        // Lᵀ is upper triangular, L is lower triangular stored row-major
        // Lᵀ[i,k] = L[k,i] = l[k * n + i]
        for i in (0..n).rev() {
            let mut sum = xi[i];
            for k in i + 1..n {
                // Lᵀ[i,k] * xi[k] = L[k,i] * xi[k]
                sum -= l[k * n + i] * xi[k];
            }
            xi[i] = sum / l[i * n + i];
        }

        // Store result in same format: x_j[i] = x[i * n + j]
        for i in 0..n {
            x[i * n + j] = xi[i];
        }
    }

    Ok(x)
}

/// Solve generalized eigenvalue problem with B = I (identity)
///
/// This is a convenience function that reduces to standard eigenvalue problem.
///
/// # Errors
///
/// Returns [`Err`] if matrix dimensions are invalid or eigendecomposition fails.
pub fn gen_eigh_identity_b(a: &[f64], n: usize) -> Result<GenEighDecomposition> {
    let eigh = eigh_f64(a, n)?;

    Ok(GenEighDecomposition {
        eigenvalues: eigh.eigenvalues,
        eigenvectors: eigh.eigenvectors,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available_sync;

    #[test]
    fn test_gen_eigh_identity_b() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        // When B = I, gen_eigh should give same results as standard eigh
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // Identity

        let result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        // Standard eigenvalues of [[4, 1], [1, 3]]
        // Characteristic polynomial: (4-λ)(3-λ) - 1 = λ² - 7λ + 11
        // λ = (7 ± √5) / 2 ≈ 2.382, 4.618
        let expected_min = (7.0 - 5.0_f64.sqrt()) / 2.0;
        let expected_max = f64::midpoint(7.0, 5.0_f64.sqrt());

        assert!(
            (result.eigenvalues[0] - expected_min).abs() < 1e-10,
            "Expected λ₀ ≈ {}, got {}",
            expected_min,
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - expected_max).abs() < 1e-10,
            "Expected λ₁ ≈ {}, got {}",
            expected_max,
            result.eigenvalues[1]
        );
    }

    #[test]
    fn test_gen_eigh_scaled_b() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        // If B = 2I, eigenvalues should be halved
        let a = vec![4.0, 0.0, 0.0, 6.0];
        let b = vec![2.0, 0.0, 0.0, 2.0]; // 2I

        let result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        // A has eigenvalues 4, 6
        // Gen. eigenvalues with B = 2I should be 4/2 = 2, 6/2 = 3
        assert!(
            (result.eigenvalues[0] - 2.0).abs() < 1e-10,
            "Expected λ₀ = 2, got {}",
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - 3.0).abs() < 1e-10,
            "Expected λ₁ = 3, got {}",
            result.eigenvalues[1]
        );
    }

    #[test]
    fn test_gen_eigh_verify() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        // Test that Ax = λBx holds
        let a = vec![5.0, 2.0, 2.0, 3.0];
        let b = vec![2.0, 1.0, 1.0, 2.0]; // SPD

        let result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        let residual = result.verify(&a, &b);
        assert!(residual < 1e-10, "Residual too large: {residual}");
    }

    #[test]
    fn test_gen_eigh_3x3() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        // 3×3 example
        #[rustfmt::skip]
        let a = vec![
            6.0, 2.0, 1.0,
            2.0, 5.0, 2.0,
            1.0, 2.0, 4.0,
        ];
        #[rustfmt::skip]
        let b = vec![
            2.0, 0.5, 0.0,
            0.5, 2.0, 0.5,
            0.0, 0.5, 2.0,
        ];

        let result = gen_eigh_f64(device, &a, &b, 3).unwrap();

        // Verify all eigenpairs
        let residual = result.verify(&a, &b);
        assert!(residual < 1e-9, "Residual too large: {residual}");
    }

    #[test]
    fn test_gen_eigh_eigenvector_access() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];

        let result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        let v0 = result.eigenvector(0).unwrap();
        let v1 = result.eigenvector(1).unwrap();

        assert_eq!(v0.len(), 2);
        assert_eq!(v1.len(), 2);

        // Out of bounds
        assert!(result.eigenvector(2).is_none());
    }

    #[test]
    fn test_gen_eigh_sort_descending() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let a = vec![1.0, 0.0, 0.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];

        let mut result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        // Initially ascending (1, 3)
        assert!(result.eigenvalues[0] < result.eigenvalues[1]);

        result.sort_descending();

        // After sorting: descending (3, 1)
        assert!(result.eigenvalues[0] > result.eigenvalues[1]);
        assert!((result.eigenvalues[0] - 3.0).abs() < 1e-10);
        assert!((result.eigenvalues[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gen_eigh_non_spd_b_error() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let a = vec![1.0, 0.0, 0.0, 1.0];
        // B is not positive definite (has negative eigenvalue)
        let b = vec![1.0, 2.0, 2.0, 1.0]; // Eigenvalues: 3, -1

        let result = gen_eigh_f64(device, &a, &b, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gen_eigh_dimension_mismatch() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0]; // Wrong size

        let result = gen_eigh_f64(device, &a, &b, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gen_eigh_trace() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let result = gen_eigh_f64(device, &a, &b, 2).unwrap();

        // With B = I, trace of eigenvalues = trace of A = 6
        let trace = result.trace();
        assert!((trace - 6.0).abs() < 1e-10, "Expected trace 6, got {trace}");
    }

    #[test]
    fn test_gen_eigh_larger_matrix() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let n = 5;
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 5.0 } else { 0.5 };
                b[i * n + j] = if i == j { 2.0 } else { 0.1 };
            }
        }
        let result = gen_eigh_f64(device, &a, &b, n).unwrap();

        let residual = result.verify(&a, &b);
        assert!(residual < 1e-8, "Residual too large for 5×5: {residual}");
    }
}
