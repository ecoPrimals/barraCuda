// SPDX-License-Identifier: AGPL-3.0-or-later
//! Singular Value Decomposition (SVD)
//!
//! Factors a matrix A into A = U Σ V^T where:
//! - U is m×m orthogonal (left singular vectors)
//! - Σ is m×n diagonal (singular values, non-negative, descending)
//! - V is n×n orthogonal (right singular vectors)
//!
//! # Algorithm
//!
//! Uses the Golub-Kahan bidiagonalization followed by implicit QR iteration.
//! This is a simplified implementation suitable for small-to-medium matrices.
//!
//! # Applications
//!
//! - Principal Component Analysis (PCA)
//! - Pseudoinverse computation
//! - Low-rank approximation
//! - Solving ill-conditioned linear systems
//! - Determining matrix rank
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations"
//! - Press et al., "Numerical Recipes"

use crate::error::{BarracudaError, Result};

/// Result of SVD decomposition A = U Σ V^T.
#[derive(Debug, Clone)]
pub struct SvdDecomposition {
    /// Left singular vectors U (m×m row-major)
    pub u: Vec<f64>,
    /// Singular values σ (min(m,n) values, descending)
    pub s: Vec<f64>,
    /// Right singular vectors V^T (n×n row-major) - note: V transpose, not V
    pub vt: Vec<f64>,
    /// Number of rows
    pub m: usize,
    /// Number of columns
    pub n: usize,
}

impl SvdDecomposition {
    /// Compute the pseudoinverse (Moore-Penrose inverse) A⁺.
    /// A⁺ = V Σ⁺ U^T where Σ⁺ has 1/σᵢ for σᵢ > tol, else 0.
    #[must_use]
    pub fn pseudoinverse(&self, tol: f64) -> Vec<f64> {
        let k = self.s.len();

        // Compute Σ⁺
        let s_inv: Vec<f64> = self
            .s
            .iter()
            .map(|&s| if s > tol { 1.0 / s } else { 0.0 })
            .collect();

        // A⁺ = V Σ⁺ U^T = (V^T)^T Σ⁺ U^T
        // A⁺ is n×m
        let mut pinv = vec![0.0; self.n * self.m];

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = 0.0;
                for l in 0..k {
                    // V[i,l] = (V^T)[l,i], U^T[l,j] = U[j,l]
                    sum += self.vt[l * self.n + i] * s_inv[l] * self.u[j * self.m + l];
                }
                pinv[i * self.m + j] = sum;
            }
        }

        pinv
    }

    /// Solve the least squares problem min‖Ax - b‖₂ using pseudoinverse.
    /// # Errors
    /// Returns [`Err`] if `b.len() != self.m` (right-hand side length must match matrix row count).
    pub fn solve(&self, b: &[f64], tol: f64) -> Result<Vec<f64>> {
        if b.len() != self.m {
            return Err(BarracudaError::InvalidInput {
                message: format!("b has length {}, expected {}", b.len(), self.m),
            });
        }

        let pinv = self.pseudoinverse(tol);

        // x = A⁺ b
        let mut x = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.m {
                x[i] += pinv[i * self.m + j] * b[j];
            }
        }

        Ok(x)
    }

    /// Compute the numerical rank (number of singular values > tol).
    #[must_use]
    pub fn rank(&self, tol: f64) -> usize {
        self.s.iter().filter(|&&s| s > tol).count()
    }

    /// Compute the condition number `σ_max` / `σ_min`.
    /// # Panics
    /// Panics if `s` is empty (should not occur after empty check).
    #[must_use]
    pub fn condition_number(&self) -> f64 {
        if self.s.is_empty() {
            return f64::INFINITY;
        }
        let s_max = self.s[0];
        let Some(&s_min) = self.s.last() else {
            return f64::INFINITY;
        };
        if s_min < 1e-14 {
            f64::INFINITY
        } else {
            s_max / s_min
        }
    }

    /// Low-rank approximation: keep only the k largest singular values.
    /// Returns the approximated matrix as m×n row-major.
    #[must_use]
    pub fn low_rank_approx(&self, k: usize) -> Vec<f64> {
        let k = k.min(self.s.len());
        let mut approx = vec![0.0; self.m * self.n];

        for r in 0..k {
            let sigma = self.s[r];
            for i in 0..self.m {
                for j in 0..self.n {
                    approx[i * self.n + j] +=
                        sigma * self.u[i * self.m + r] * self.vt[r * self.n + j];
                }
            }
        }

        approx
    }
}

/// Compute SVD using a one-sided Jacobi algorithm.
///
/// This is a simple, robust method suitable for small matrices.
/// For large matrices, use bidiagonalization + QR iteration.
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
/// use barracuda::ops::linalg::svd_decompose;
///
/// let a = vec![
///     1.0, 2.0,
///     3.0, 4.0,
///     5.0, 6.0,
/// ];
/// let svd = svd_decompose(&a, 3, 2).unwrap();
/// println!("Singular values: {:?}", svd.s);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn svd_decompose(a: &[f64], m: usize, n: usize) -> Result<SvdDecomposition> {
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

    // For simplicity, we compute A^T A eigenvalues and eigenvectors
    // This gives V and σ². Then U = A V Σ^(-1).
    //
    // Note: This approach works well for n ≤ m. For m < n, we'd compute A A^T instead.

    // Compute A^T A (n×n)
    let mut ata = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[k * n + i] * a[k * n + j];
            }
            ata[i * n + j] = sum;
        }
    }

    // Compute eigenvalues/eigenvectors of A^T A using Jacobi iteration
    let (eigenvalues, eigenvectors) = jacobi_eigen(&ata, n, 100)?;

    // Singular values are sqrt(eigenvalues of A^T A)
    // Sort in descending order
    let mut indexed: Vec<(usize, f64)> = eigenvalues
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e.max(0.0).sqrt()))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = n.min(m); // Number of singular values
    let s: Vec<f64> = indexed.iter().take(k).map(|(_, v)| *v).collect();

    // V^T: rows are right singular vectors in sorted order
    let mut vt = vec![0.0; n * n];
    for i in 0..n {
        let orig_idx = indexed[i].0;
        for j in 0..n {
            vt[i * n + j] = eigenvectors[j * n + orig_idx];
        }
    }

    // U = A V Σ^(-1)
    // U is m×m, but we only compute the first k columns from AV
    let mut u = vec![0.0; m * m];

    // First compute AV (m×n)
    let mut av = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..n {
                // V[l,j] = V^T[j,l]
                sum += a[i * n + l] * vt[j * n + l];
            }
            av[i * n + j] = sum;
        }
    }

    // U[:,j] = AV[:,j] / σ_j
    for j in 0..k {
        if s[j] > 1e-14 {
            for i in 0..m {
                u[i * m + j] = av[i * n + j] / s[j];
            }
        }
    }

    // Complete U to be orthogonal (Gram-Schmidt for remaining columns)
    for j in k..m {
        // Start with a standard basis vector
        let mut col = vec![0.0; m];
        col[j % m] = 1.0;

        // Orthogonalize against existing columns
        for prev in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += u[i * m + prev] * col[i];
            }
            for i in 0..m {
                col[i] -= dot * u[i * m + prev];
            }
        }

        // Normalize
        let mut norm = 0.0;
        for i in 0..m {
            norm += col[i] * col[i];
        }
        norm = norm.sqrt();

        if norm > 1e-14 {
            for i in 0..m {
                u[i * m + j] = col[i] / norm;
            }
        }
    }

    Ok(SvdDecomposition { u, s, vt, m, n })
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors is n×n column-major.
fn jacobi_eigen(a: &[f64], n: usize, max_iter: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];

    // Initialize V as identity
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_val < 1e-14 {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (aqq - app).abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4 * apq.signum()
        } else {
            0.5 * (2.0 * apq / (aqq - app)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A: A' = J^T A J
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                a[i * n + p] = c * aip - s * aiq;
                a[p * n + i] = a[i * n + p];
                a[i * n + q] = s * aip + c * aiq;
                a[q * n + i] = a[i * n + q];
            }
        }

        a[p * n + p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Apply rotation to V: V' = V J
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    Ok((eigenvalues, v))
}

/// Convenience function: compute singular values only.
///
/// # Errors
///
/// Returns [`Err`] if [`svd_decompose`] fails (invalid dimensions or element count).
pub fn svd_values(a: &[f64], m: usize, n: usize) -> Result<Vec<f64>> {
    let svd = svd_decompose(a, m, n)?;
    Ok(svd.s)
}

/// Convenience function: compute pseudoinverse.
///
/// # Errors
///
/// Returns [`Err`] if [`svd_decompose`] fails (invalid dimensions or element count).
pub fn svd_pinv(a: &[f64], m: usize, n: usize, tol: f64) -> Result<Vec<f64>> {
    let svd = svd_decompose(a, m, n)?;
    Ok(svd.pseudoinverse(tol))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_svd_2x2() {
        let a = vec![3.0, 2.0, 2.0, 3.0];
        let svd = svd_decompose(&a, 2, 2).unwrap();

        // Singular values of [[3,2],[2,3]] are 5 and 1
        assert!(approx_eq(svd.s[0], 5.0, 1e-10), "s[0] = {}", svd.s[0]);
        assert!(approx_eq(svd.s[1], 1.0, 1e-10), "s[1] = {}", svd.s[1]);

        // Verify U Σ V^T = A
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += svd.u[i * 2 + k] * svd.s[k] * svd.vt[k * 2 + j];
                }
                assert!(
                    approx_eq(val, a[i * 2 + j], 1e-10),
                    "UΣV^T[{},{}] = {}, A = {}",
                    i,
                    j,
                    val,
                    a[i * 2 + j]
                );
            }
        }
    }

    #[test]
    fn test_svd_rank_deficient() {
        // Rank-1 matrix
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let svd = svd_decompose(&a, 2, 2).unwrap();

        // One singular value should be ~0
        assert!(svd.s[1] < 1e-10, "s[1] = {}", svd.s[1]);

        // Rank should be 1
        assert_eq!(svd.rank(1e-10), 1);
    }

    #[test]
    fn test_svd_3x2() {
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let svd = svd_decompose(&a, 3, 2).unwrap();

        // Should have 2 non-zero singular values
        assert_eq!(svd.s.len(), 2);
        assert!(svd.s[0] > 0.0);
        assert!(svd.s[1] > 0.0);

        // Verify reconstruction
        let approx = svd.low_rank_approx(2);
        for i in 0..6 {
            assert!(
                approx_eq(approx[i], a[i], 1e-10),
                "approx[{}] = {}, a = {}",
                i,
                approx[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_svd_pseudoinverse() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let svd = svd_decompose(&a, 2, 2).unwrap();
        let pinv = svd.pseudoinverse(1e-10);

        // A * A⁺ * A should equal A
        // First compute A * A⁺
        let mut aa_pinv = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    aa_pinv[i * 2 + j] += a[i * 2 + k] * pinv[k * 2 + j];
                }
            }
        }

        // Then compute (A * A⁺) * A
        let mut result = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i * 2 + j] += aa_pinv[i * 2 + k] * a[k * 2 + j];
                }
            }
        }

        for i in 0..4 {
            assert!(
                approx_eq(result[i], a[i], 1e-10),
                "A*A⁺*A[{}] = {}, A = {}",
                i,
                result[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_svd_condition_number() {
        // Well-conditioned matrix
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let svd = svd_decompose(&a, 2, 2).unwrap();
        assert!(approx_eq(svd.condition_number(), 1.0, 1e-10));

        // Ill-conditioned matrix
        let b = vec![1.0, 0.0, 0.0, 1e-10];
        let svd_b = svd_decompose(&b, 2, 2).unwrap();
        assert!(svd_b.condition_number() > 1e9);
    }

    #[test]
    fn test_svd_low_rank_approx() {
        // 3x3 matrix with rank 2
        let a = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
        let svd = svd_decompose(&a, 3, 3).unwrap();

        // Rank-1 approximation
        let approx1 = svd.low_rank_approx(1);

        // Should be close to original since matrix is rank-1
        for i in 0..9 {
            assert!(
                approx_eq(approx1[i], a[i], 1e-10),
                "approx1[{}] = {}, a = {}",
                i,
                approx1[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_svd_solve() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0];
        let svd = svd_decompose(&a, 3, 2).unwrap();
        let x = svd.solve(&b, 1e-10).unwrap();

        // Verify this is a least squares solution
        // The residual r = b - Ax should be orthogonal to column space of A
        let mut ax = [0.0; 3];
        for i in 0..3 {
            for j in 0..2 {
                ax[i] += a[i * 2 + j] * x[j];
            }
        }

        // Compute A^T * (b - Ax)
        let r: Vec<f64> = (0..3).map(|i| b[i] - ax[i]).collect();
        let mut atr = [0.0; 2];
        for j in 0..2 {
            for i in 0..3 {
                atr[j] += a[i * 2 + j] * r[i];
            }
        }

        // Should be ~0
        for j in 0..2 {
            assert!(atr[j].abs() < 1e-10, "A^T r[{}] = {}", j, atr[j]);
        }
    }

    #[test]
    fn test_svd_errors() {
        assert!(svd_decompose(&[], 0, 0).is_err());
        assert!(svd_decompose(&[1.0, 2.0, 3.0], 2, 2).is_err());
    }
}
