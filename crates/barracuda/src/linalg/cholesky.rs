// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cholesky Decomposition (f64)
//!
//! Computes the Cholesky decomposition of a symmetric positive-definite matrix:
//! A = L·Lᵀ where L is lower triangular.
//!
//! # Algorithm
//!
//! GPU-accelerated via CholeskyF64 (WGSL). CPU fallback for tests only.
//!
//! # Applications
//!
//! - RBF surrogate kernel matrix solve
//! - Gaussian process covariance
//! - Sampling from multivariate normal distributions
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations", Algorithm 4.2.1

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// Result of Cholesky decomposition A = L·Lᵀ.
#[derive(Debug, Clone)]
pub struct CholeskyDecomposition {
    /// Lower triangular matrix L (n×n, row-major)
    pub l: Vec<f64>,
    /// Matrix dimension
    pub n: usize,
}

impl CholeskyDecomposition {
    /// Solve the linear system Ax = b.
    ///
    /// Uses forward substitution (Ly = b) then backward substitution (Lᵀx = y).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        if b.len() != self.n {
            return Err(BarracudaError::InvalidInput {
                message: format!("b has length {}, expected {}", b.len(), self.n),
            });
        }

        // Forward substitution: Ly = b
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.l[i * self.n + j] * y[j];
            }
            let diag = self.l[i * self.n + i];
            if diag.abs() < 1e-14 {
                return Err(BarracudaError::Numerical {
                    message: "Matrix is singular or not positive definite".to_string(),
                });
            }
            y[i] = sum / diag;
        }

        // Backward substitution: Lᵀx = y
        let mut x = vec![0.0; self.n];
        for i in (0..self.n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..self.n {
                // Lᵀ[i,j] = L[j,i]
                sum -= self.l[j * self.n + i] * x[j];
            }
            x[i] = sum / self.l[i * self.n + i];
        }

        Ok(x)
    }

    /// Compute the determinant: det(A) = det(L)² = (∏ L[i,i])²
    pub fn det(&self) -> f64 {
        let mut det_l = 1.0;
        for i in 0..self.n {
            det_l *= self.l[i * self.n + i];
        }
        det_l * det_l
    }

    /// Compute log determinant: log(det(A)) = 2·∑ log(L[i,i])
    ///
    /// More numerically stable for large matrices.
    pub fn log_det(&self) -> f64 {
        let mut log_det = 0.0;
        for i in 0..self.n {
            log_det += self.l[i * self.n + i].ln();
        }
        2.0 * log_det
    }

    /// Compute the inverse of A via L⁻¹.
    pub fn inverse(&self) -> Result<Vec<f64>> {
        let mut inv = vec![0.0; self.n * self.n];

        // Solve A·X = I column by column
        for j in 0..self.n {
            let mut e = vec![0.0; self.n];
            e[j] = 1.0;
            let col = self.solve(&e)?;
            for i in 0..self.n {
                inv[i * self.n + j] = col[i];
            }
        }

        Ok(inv)
    }
}

/// Compute Cholesky decomposition of a symmetric positive-definite matrix (GPU).
///
/// Returns L such that A = L·Lᵀ.
///
/// # Arguments
///
/// * `device` - GPU device for execution
/// * `a` - n×n symmetric positive-definite matrix (row-major)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// `CholeskyDecomposition` containing lower triangular L
///
/// # Errors
///
/// Returns error if matrix dimensions are invalid.
pub fn cholesky_f64(device: Arc<WgpuDevice>, a: &[f64], n: usize) -> Result<CholeskyDecomposition> {
    if a.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Matrix has {} elements, expected {}×{}", a.len(), n, n),
        });
    }

    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Matrix dimension must be positive".to_string(),
        });
    }

    let l = crate::ops::linalg::cholesky::CholeskyF64::execute(device, a, n)?;

    // Validate SPD: Cholesky of a non-positive-definite matrix produces NaN
    // on the diagonal (sqrt of negative value). Detect and report this.
    for i in 0..n {
        let diag = l[i * n + i];
        if diag.is_nan() || diag <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("Matrix is not positive definite (L[{i},{i}] = {diag})"),
            });
        }
    }

    Ok(CholeskyDecomposition { l, n })
}

/// Compute Cholesky decomposition on CPU (test/benchmark only).
#[cfg(any(test, feature = "benchmarks"))]
pub fn cholesky_f64_cpu(a: &[f64], n: usize) -> Result<CholeskyDecomposition> {
    if a.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Matrix has {} elements, expected {}×{}", a.len(), n, n),
        });
    }

    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Matrix dimension must be positive".to_string(),
        });
    }

    let mut l = vec![0.0; n * n];

    // Cholesky-Banachiewicz algorithm
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];

            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }

            if i == j {
                // Diagonal element
                if sum <= 0.0 {
                    return Err(BarracudaError::Numerical {
                        message: format!(
                            "Matrix is not positive definite: L[{},{}]² = {} ≤ 0",
                            i, i, sum
                        ),
                    });
                }
                l[i * n + j] = sum.sqrt();
            } else {
                // Off-diagonal element
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }

    Ok(CholeskyDecomposition { l, n })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cholesky_2x2() {
        // A = [[4, 2], [2, 3]]
        // L = [[2, 0], [1, √2]]
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();

        assert!(approx_eq(chol.l[0], 2.0, 1e-10)); // L[0,0]
        assert!(approx_eq(chol.l[1], 0.0, 1e-10)); // L[0,1]
        assert!(approx_eq(chol.l[2], 1.0, 1e-10)); // L[1,0]
        assert!(approx_eq(chol.l[3], std::f64::consts::SQRT_2, 1e-10)); // L[1,1]
    }

    #[test]
    fn test_cholesky_solve() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![10.0, 8.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();
        let x = chol.solve(&b).unwrap();

        // Verify Ax = b
        let ax0 = 4.0 * x[0] + 2.0 * x[1];
        let ax1 = 2.0 * x[0] + 3.0 * x[1];
        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
    }

    #[test]
    fn test_cholesky_det() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();

        // det(A) = 4*3 - 2*2 = 8
        assert!(approx_eq(chol.det(), 8.0, 1e-10));
    }

    #[test]
    fn test_cholesky_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();

        // L = I
        assert!(approx_eq(chol.l[0], 1.0, 1e-10));
        assert!(approx_eq(chol.l[1], 0.0, 1e-10));
        assert!(approx_eq(chol.l[2], 0.0, 1e-10));
        assert!(approx_eq(chol.l[3], 1.0, 1e-10));
    }

    #[test]
    fn test_cholesky_3x3() {
        // 3×3 SPD matrix
        let a = vec![4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0];
        let chol = cholesky_f64_cpu(&a, 3).unwrap();

        // Verify L·Lᵀ = A
        let mut llt = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    llt[i * 3 + j] += chol.l[i * 3 + k] * chol.l[j * 3 + k];
                }
            }
        }

        for i in 0..9 {
            assert!(
                approx_eq(llt[i], a[i], 1e-10),
                "L·Lᵀ[{}] = {}, A = {}",
                i,
                llt[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_cholesky_inverse() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();
        let inv = chol.inverse().unwrap();

        // A * A⁻¹ = I
        let mut aa_inv = vec![0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    aa_inv[i * 2 + j] += a[i * 2 + k] * inv[k * 2 + j];
                }
            }
        }

        assert!(approx_eq(aa_inv[0], 1.0, 1e-10));
        assert!(approx_eq(aa_inv[1], 0.0, 1e-10));
        assert!(approx_eq(aa_inv[2], 0.0, 1e-10));
        assert!(approx_eq(aa_inv[3], 1.0, 1e-10));
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite (eigenvalues: 4, -1)
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky_f64_cpu(&a, 2).is_err());
    }

    #[test]
    fn test_cholesky_log_det() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let chol = cholesky_f64_cpu(&a, 2).unwrap();

        // log(det(A)) = log(8) ≈ 2.079
        let expected = 8.0_f64.ln();
        assert!(approx_eq(chol.log_det(), expected, 1e-10));
    }

    #[test]
    fn test_cholesky_errors() {
        assert!(cholesky_f64_cpu(&[], 0).is_err());
        assert!(cholesky_f64_cpu(&[1.0, 2.0, 3.0], 2).is_err());
    }
}
