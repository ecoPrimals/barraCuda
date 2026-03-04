// SPDX-License-Identifier: AGPL-3.0-or-later
//! LU Decomposition
//!
//! Factors a matrix A into the product of a lower triangular matrix L
//! and an upper triangular matrix U, with partial pivoting: PA = LU.
//!
//! # Algorithm
//!
//! Uses Doolittle's method with partial pivoting for numerical stability.
//!
//! # Applications
//!
//! - Solving linear systems Ax = b
//! - Computing determinants
//! - Matrix inversion
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations"
//! - Press et al., "Numerical Recipes"

use crate::error::{BarracudaError, Result};

/// Result of LU decomposition with partial pivoting.
///
/// For input matrix A (n×n), returns L, U, P such that PA = LU.
#[derive(Debug, Clone)]
pub struct LuDecomposition {
    /// Combined L and U matrices (L has 1s on diagonal, stored below diagonal)
    /// U is stored on and above diagonal
    pub lu: Vec<f64>,
    /// Permutation vector (P[i] = j means row i was swapped with row j)
    pub perm: Vec<usize>,
    /// Matrix dimension
    pub n: usize,
    /// Number of row swaps (for determinant sign)
    pub num_swaps: usize,
}

impl LuDecomposition {
    /// Extract the L matrix (lower triangular with 1s on diagonal).
    pub fn l(&self) -> Vec<f64> {
        let mut l = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            l[i * self.n + i] = 1.0; // Diagonal is 1
            for j in 0..i {
                l[i * self.n + j] = self.lu[i * self.n + j];
            }
        }
        l
    }

    /// Extract the U matrix (upper triangular).
    pub fn u(&self) -> Vec<f64> {
        let mut u = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            for j in i..self.n {
                u[i * self.n + j] = self.lu[i * self.n + j];
            }
        }
        u
    }

    /// Get the permutation matrix P as a dense matrix.
    pub fn p(&self) -> Vec<f64> {
        let mut p = vec![0.0; self.n * self.n];
        for i in 0..self.n {
            p[i * self.n + self.perm[i]] = 1.0;
        }
        p
    }

    /// Compute the determinant of the original matrix.
    ///
    /// det(A) = (-1)^swaps × ∏ U[i,i]
    #[expect(
        clippy::manual_is_multiple_of,
        reason = "is_multiple_of is nightly-only"
    )]
    pub fn det(&self) -> f64 {
        let mut det = if self.num_swaps % 2 == 0 { 1.0 } else { -1.0 };
        for i in 0..self.n {
            det *= self.lu[i * self.n + i];
        }
        det
    }

    /// Solve the linear system Ax = b.
    ///
    /// First solves Ly = Pb (forward substitution), then Ux = y (back substitution).
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>> {
        if b.len() != self.n {
            return Err(BarracudaError::InvalidInput {
                message: format!("b has length {}, expected {}", b.len(), self.n),
            });
        }

        // Apply permutation to b
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            y[i] = b[self.perm[i]];
        }

        // Forward substitution: Ly = Pb
        for i in 1..self.n {
            for j in 0..i {
                y[i] -= self.lu[i * self.n + j] * y[j];
            }
        }

        // Back substitution: Ux = y
        let mut x = y;
        for i in (0..self.n).rev() {
            for j in (i + 1)..self.n {
                x[i] -= self.lu[i * self.n + j] * x[j];
            }
            if self.lu[i * self.n + i].abs() < 1e-14 {
                return Err(BarracudaError::Numerical {
                    message: "Matrix is singular".to_string(),
                });
            }
            x[i] /= self.lu[i * self.n + i];
        }

        Ok(x)
    }

    /// Compute the inverse of the original matrix.
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

/// Compute LU decomposition with partial pivoting.
///
/// Factors A (n×n row-major) into PA = LU where:
/// - P is a permutation matrix
/// - L is lower triangular with 1s on diagonal
/// - U is upper triangular
///
/// # Arguments
///
/// * `a` - n×n matrix in row-major order
/// * `n` - Matrix dimension
///
/// # Example
///
/// ```
/// use barracuda::ops::linalg::lu_decompose;
///
/// let a = vec![
///     4.0, 3.0,
///     6.0, 3.0,
/// ];
/// let lu = lu_decompose(&a, 2).unwrap();
///
/// // Solve Ax = b
/// let b = vec![10.0, 12.0];
/// let x = lu.solve(&b).unwrap();
/// ```
pub fn lu_decompose(a: &[f64], n: usize) -> Result<LuDecomposition> {
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

    // Copy matrix
    let mut lu = a.to_vec();

    // Initialize permutation
    let mut perm: Vec<usize> = (0..n).collect();
    let mut num_swaps = 0;

    // Doolittle's algorithm with partial pivoting
    for k in 0..n {
        // Find pivot (largest element in column k, rows k..n)
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;

        for i in (k + 1)..n {
            let val = lu[i * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Swap rows if needed
        if max_row != k {
            // Swap rows in LU
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            // Swap in permutation
            perm.swap(k, max_row);
            num_swaps += 1;
        }

        // Check for singularity
        let pivot = lu[k * n + k];
        if pivot.abs() < 1e-14 {
            // Matrix is singular or nearly singular
            // We continue but determinant will be ~0
        }

        // Elimination
        for i in (k + 1)..n {
            if pivot.abs() > 1e-14 {
                lu[i * n + k] /= pivot;
            }
            for j in (k + 1)..n {
                lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
            }
        }
    }

    Ok(LuDecomposition {
        lu,
        perm,
        n,
        num_swaps,
    })
}

/// Convenience function to solve Ax = b using LU decomposition.
pub fn lu_solve(a: &[f64], n: usize, b: &[f64]) -> Result<Vec<f64>> {
    let lu = lu_decompose(a, n)?;
    lu.solve(b)
}

/// Compute the determinant of a square matrix using LU decomposition.
pub fn lu_det(a: &[f64], n: usize) -> Result<f64> {
    let lu = lu_decompose(a, n)?;
    Ok(lu.det())
}

/// Compute the inverse of a square matrix using LU decomposition.
pub fn lu_inverse(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let lu = lu_decompose(a, n)?;
    lu.inverse()
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_lu_2x2() {
        let a = vec![4.0, 3.0, 6.0, 3.0];
        let lu = lu_decompose(&a, 2).unwrap();

        // Check determinant: det = 4*3 - 3*6 = -6
        assert!(approx_eq(lu.det(), -6.0, 1e-10));
    }

    #[test]
    fn test_lu_solve_2x2() {
        let a = vec![4.0, 3.0, 6.0, 3.0];
        let b = vec![10.0, 12.0];
        let x = lu_solve(&a, 2, &b).unwrap();

        // Verify: Ax = b
        let ax0 = 4.0 * x[0] + 3.0 * x[1];
        let ax1 = 6.0 * x[0] + 3.0 * x[1];
        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
    }

    #[test]
    fn test_lu_solve_3x3() {
        let a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let b = vec![1.0, 0.0, 1.0];
        let x = lu_solve(&a, 3, &b).unwrap();

        // Verify: Ax = b
        let ax0 = 2.0 * x[0] - x[1];
        let ax1 = -x[0] + 2.0 * x[1] - x[2];
        let ax2 = -x[1] + 2.0 * x[2];

        assert!(approx_eq(ax0, b[0], 1e-10));
        assert!(approx_eq(ax1, b[1], 1e-10));
        assert!(approx_eq(ax2, b[2], 1e-10));
    }

    #[test]
    fn test_lu_det_3x3() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let det = lu_det(&a, 3).unwrap();

        // det = 1*(5*10 - 6*8) - 2*(4*10 - 6*7) + 3*(4*8 - 5*7)
        //     = 1*(50-48) - 2*(40-42) + 3*(32-35)
        //     = 2 + 4 - 9 = -3
        assert!(approx_eq(det, -3.0, 1e-10));
    }

    #[test]
    fn test_lu_inverse_2x2() {
        let a = vec![4.0, 7.0, 2.0, 6.0];
        let inv = lu_inverse(&a, 2).unwrap();

        // A * A^(-1) should be identity
        let i00 = a[0] * inv[0] + a[1] * inv[2];
        let i01 = a[0] * inv[1] + a[1] * inv[3];
        let i10 = a[2] * inv[0] + a[3] * inv[2];
        let i11 = a[2] * inv[1] + a[3] * inv[3];

        assert!(approx_eq(i00, 1.0, 1e-10));
        assert!(approx_eq(i01, 0.0, 1e-10));
        assert!(approx_eq(i10, 0.0, 1e-10));
        assert!(approx_eq(i11, 1.0, 1e-10));
    }

    #[test]
    fn test_lu_identity() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let lu = lu_decompose(&a, 3).unwrap();

        assert!(approx_eq(lu.det(), 1.0, 1e-10));

        let l = lu.l();
        let u = lu.u();

        // L should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(l[i * 3 + j], expected, 1e-10));
            }
        }

        // U should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(u[i * 3 + j], expected, 1e-10));
            }
        }
    }

    #[test]
    fn test_lu_needs_pivoting() {
        // Matrix where first pivot is zero
        let a = vec![0.0, 1.0, 1.0, 1.0];
        let lu = lu_decompose(&a, 2).unwrap();

        // Should still work with pivoting
        assert!(approx_eq(lu.det(), -1.0, 1e-10));

        let x = lu.solve(&[3.0, 4.0]).unwrap();
        // Verify
        let ax0 = 0.0 * x[0] + 1.0 * x[1];
        let ax1 = 1.0 * x[0] + 1.0 * x[1];
        assert!(approx_eq(ax0, 3.0, 1e-10));
        assert!(approx_eq(ax1, 4.0, 1e-10));
    }

    #[test]
    fn test_lu_empty_error() {
        assert!(lu_decompose(&[], 0).is_err());
    }

    #[test]
    fn test_lu_size_mismatch_error() {
        assert!(lu_decompose(&[1.0, 2.0, 3.0], 2).is_err());
    }
}
