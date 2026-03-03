// SPDX-License-Identifier: AGPL-3.0-or-later
//! QR Decomposition
//!
//! Factors a matrix A into the product of an orthogonal matrix Q and
//! an upper triangular matrix R: A = QR.
//!
//! # Algorithm
//!
//! Uses Householder reflections for numerical stability.
//!
//! # Applications
//!
//! - Least squares problems
//! - Eigenvalue computation (QR algorithm)
//! - Orthogonalization
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations"
//! - Trefethen & Bau, "Numerical Linear Algebra"

use crate::error::{BarracudaError, Result};

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QrDecomposition {
    /// Orthogonal matrix Q (m×m, row-major)
    pub q: Vec<f64>,
    /// Upper triangular matrix R (m×n, row-major)
    pub r: Vec<f64>,
    /// Number of rows
    pub m: usize,
    /// Number of columns
    pub n: usize,
}

impl QrDecomposition {
    /// Solve the least squares problem min‖Ax - b‖₂.
    ///
    /// For overdetermined systems (m > n), finds the least squares solution.
    /// For square systems, equivalent to direct solve.
    pub fn solve_least_squares(&self, b: &[f64]) -> Result<Vec<f64>> {
        if b.len() != self.m {
            return Err(BarracudaError::InvalidInput {
                message: format!("b has length {}, expected {}", b.len(), self.m),
            });
        }

        // Compute Q^T * b
        let mut qtb = vec![0.0; self.m];
        for i in 0..self.m {
            for j in 0..self.m {
                qtb[i] += self.q[j * self.m + i] * b[j]; // Q^T
            }
        }

        // Back substitution: Rx = Q^T b (use first n components)
        let mut x = vec![0.0; self.n];
        for i in (0..self.n).rev() {
            let mut sum = qtb[i];
            for j in (i + 1)..self.n {
                sum -= self.r[i * self.n + j] * x[j];
            }
            let diag = self.r[i * self.n + i];
            if diag.abs() < 1e-14 {
                return Err(BarracudaError::Numerical {
                    message: "Matrix is rank deficient".to_string(),
                });
            }
            x[i] = sum / diag;
        }

        Ok(x)
    }
}

/// Compute QR decomposition using Householder reflections.
///
/// Factors A (m×n row-major, m ≥ n) into A = QR where:
/// - Q is m×m orthogonal (Q^T Q = I)
/// - R is m×n upper triangular
///
/// # Arguments
///
/// * `a` - m×n matrix in row-major order
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Example
///
/// ```
/// use barracuda::ops::linalg::qr_decompose;
///
/// let a = vec![
///     1.0, 1.0,
///     1.0, 2.0,
///     1.0, 3.0,
/// ];
/// let qr = qr_decompose(&a, 3, 2).unwrap();
///
/// // Solve least squares: min‖Ax - b‖
/// let b = vec![1.0, 2.0, 2.0];
/// let x = qr.solve_least_squares(&b).unwrap();
/// ```
pub fn qr_decompose(a: &[f64], m: usize, n: usize) -> Result<QrDecomposition> {
    if a.len() != m * n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Matrix has {} elements, expected {}×{}", a.len(), m, n),
        });
    }

    if m == 0 || n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "Matrix dimensions must be positive".to_string(),
        });
    }

    if m < n {
        return Err(BarracudaError::InvalidInput {
            message: format!("QR requires m ≥ n, got m={m}, n={n}"),
        });
    }

    // Initialize Q as identity, R as copy of A
    let mut q = vec![0.0; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    let mut r = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            r[i * n + j] = a[i * n + j];
        }
    }

    // Householder QR
    for k in 0..n {
        // Compute Householder vector for column k
        let mut norm_sq = 0.0;
        for i in k..m {
            norm_sq += r[i * n + k] * r[i * n + k];
        }
        let norm = norm_sq.sqrt();

        if norm < 1e-14 {
            continue; // Skip zero columns
        }

        // Sign to avoid cancellation
        let sign = if r[k * n + k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = sign * norm;

        // Householder vector v
        let mut v = vec![0.0; m];
        for i in k..m {
            v[i] = r[i * n + k];
        }
        v[k] += alpha;

        // Normalize v
        let mut v_norm_sq = 0.0;
        for i in k..m {
            v_norm_sq += v[i] * v[i];
        }

        if v_norm_sq < 1e-28 {
            continue;
        }

        // Apply Householder reflection to R: R = (I - 2vv^T/‖v‖²) R
        let beta = 2.0 / v_norm_sq;

        // Apply to each column j ≥ k of R
        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * r[i * n + j];
            }
            dot *= beta;
            for i in k..m {
                r[i * n + j] -= dot * v[i];
            }
        }

        // Apply to Q: Q = Q (I - 2vv^T/‖v‖²) = Q - 2 (Qv) v^T / ‖v‖²
        for i in 0..m {
            let mut dot = 0.0;
            for j in k..m {
                dot += q[i * m + j] * v[j];
            }
            dot *= beta;
            for j in k..m {
                q[i * m + j] -= dot * v[j];
            }
        }
    }

    // Clean up small values in R below diagonal
    for i in 0..m {
        for j in 0..n {
            if i > j {
                r[i * n + j] = 0.0;
            } else if r[i * n + j].abs() < 1e-14 {
                r[i * n + j] = 0.0;
            }
        }
    }

    Ok(QrDecomposition { q, r, m, n })
}

/// Convenience function: solve least squares min‖Ax - b‖₂.
pub fn qr_least_squares(a: &[f64], m: usize, n: usize, b: &[f64]) -> Result<Vec<f64>> {
    let qr = qr_decompose(a, m, n)?;
    qr.solve_least_squares(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_qr_2x2() {
        let a = vec![3.0, 4.0, 4.0, 3.0];
        let qr = qr_decompose(&a, 2, 2).unwrap();

        // Verify Q is orthogonal: Q^T Q = I
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += qr.q[k * 2 + i] * qr.q[k * 2 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^TQ[{},{}] = {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Verify QR = A
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += qr.q[i * 2 + k] * qr.r[k * 2 + j];
                }
                assert!(
                    approx_eq(val, a[i * 2 + j], 1e-10),
                    "QR[{},{}] = {}, A = {}",
                    i,
                    j,
                    val,
                    a[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn test_qr_3x2_least_squares() {
        // Overdetermined system: 3 equations, 2 unknowns
        let a = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let b = vec![1.0, 2.0, 2.0];

        let x = qr_least_squares(&a, 3, 2, &b).unwrap();

        // Verify this is a least squares solution by checking normal equations
        // A^T A x = A^T b
        // A^T = [[1,1,1], [1,2,3]]
        // A^T A = [[3, 6], [6, 14]]
        // A^T b = [5, 11]

        let ata = vec![3.0, 6.0, 6.0, 14.0];
        let atb = vec![5.0, 11.0];

        let lhs0 = ata[0] * x[0] + ata[1] * x[1];
        let lhs1 = ata[2] * x[0] + ata[3] * x[1];

        assert!(
            approx_eq(lhs0, atb[0], 1e-10),
            "normal eq 0: {} vs {}",
            lhs0,
            atb[0]
        );
        assert!(
            approx_eq(lhs1, atb[1], 1e-10),
            "normal eq 1: {} vs {}",
            lhs1,
            atb[1]
        );
    }

    #[test]
    fn test_qr_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let qr = qr_decompose(&a, 2, 2).unwrap();

        // Q should be identity (possibly with sign flips)
        // R should be identity (possibly with sign flips)
        // QR should equal A

        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += qr.q[i * 2 + k] * qr.r[k * 2 + j];
                }
                assert!(approx_eq(val, a[i * 2 + j], 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_tall_matrix() {
        let a = vec![1.0, 2.0, 3.0];
        let qr = qr_decompose(&a, 3, 1).unwrap();

        // R should be 3×1 with only R[0,0] non-zero
        // R[0,0] should be ±‖a‖
        let norm_a = (1.0f64 + 4.0 + 9.0).sqrt();
        assert!(approx_eq(qr.r[0].abs(), norm_a, 1e-10));

        // Q should be orthogonal
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += qr.q[k * 3 + i] * qr.q[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(dot, expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_qr_solve_square() {
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 7.0];
        let x = qr_least_squares(&a, 2, 2, &b).unwrap();

        // Verify Ax = b
        let ax0 = 2.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 3.0 * x[1];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
    }

    #[test]
    fn test_qr_errors() {
        // m < n
        assert!(qr_decompose(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).is_err());

        // Wrong size
        assert!(qr_decompose(&[1.0, 2.0, 3.0], 2, 2).is_err());

        // Empty
        assert!(qr_decompose(&[], 0, 0).is_err());
    }
}
