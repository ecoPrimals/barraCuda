// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symmetric Eigenvalue Decomposition (f64 CPU)
//!
//! Computes eigenvalues and eigenvectors of a symmetric matrix:
//! A = V·D·Vᵀ where D is diagonal (eigenvalues) and V is orthogonal (eigenvectors).
//!
//! # Algorithm
//!
//! Uses the Jacobi eigenvalue algorithm with Givens rotations.
//! Converges quadratically for distinct eigenvalues.
//!
//! # Applications
//!
//! - Principal Component Analysis (PCA)
//! - Spectral clustering
//! - Vibration analysis
//! - Quantum mechanics (Hamiltonian diagonalization)
//! - hotSpring HFB nuclear physics solver
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations", Section 8.4
//! - scipy.linalg.eigh

use crate::error::{BarracudaError, Result};

/// Result of symmetric eigenvalue decomposition A = V·D·Vᵀ.
#[derive(Debug, Clone)]
pub struct EighDecomposition {
    /// Eigenvalues (n elements, sorted ascending by default)
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of V (n×n, row-major)
    /// Column j is the eigenvector for eigenvalue j
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension
    pub n: usize,
}

impl EighDecomposition {
    /// Get eigenvector for eigenvalue at index i.
    #[must_use]
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

    /// Sort eigenvalues and eigenvectors in descending order.
    /// Useful for PCA where largest eigenvalues matter most.
    pub fn sort_descending(&mut self) {
        let mut indexed: Vec<(usize, f64)> = self.eigenvalues.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let old_vals = self.eigenvalues.clone();
        let old_vecs = self.eigenvectors.clone();

        for (new_idx, (old_idx, _)) in indexed.iter().enumerate() {
            self.eigenvalues[new_idx] = old_vals[*old_idx];
            for row in 0..self.n {
                self.eigenvectors[row * self.n + new_idx] = old_vecs[row * self.n + old_idx];
            }
        }
    }

    /// Reconstruct A = V·D·Vᵀ.
    /// Useful for verification.
    #[must_use]
    pub fn reconstruct(&self) -> Vec<f64> {
        let mut a = vec![0.0; self.n * self.n];

        for i in 0..self.n {
            for j in 0..self.n {
                let mut sum = 0.0;
                for k in 0..self.n {
                    // A[i,j] = Σₖ V[i,k] * D[k,k] * V[j,k]
                    let vik = self.eigenvectors[i * self.n + k];
                    let vjk = self.eigenvectors[j * self.n + k];
                    sum += vik * self.eigenvalues[k] * vjk;
                }
                a[i * self.n + j] = sum;
            }
        }

        a
    }

    /// Compute trace (sum of eigenvalues).
    #[must_use]
    pub fn trace(&self) -> f64 {
        self.eigenvalues.iter().sum()
    }

    /// Compute determinant (product of eigenvalues).
    #[must_use]
    pub fn det(&self) -> f64 {
        self.eigenvalues.iter().product()
    }
}

/// Compute eigenvalue decomposition of a symmetric matrix.
///
/// Uses the Jacobi eigenvalue algorithm with Givens rotations.
/// Returns eigenvalues in ascending order by default.
///
/// # Arguments
///
/// * `a` - n×n symmetric matrix (row-major)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// `EighDecomposition` containing eigenvalues and eigenvectors
///
/// # Errors
///
/// Returns error if matrix dimensions are invalid.
///
/// # Example
///
/// ```
/// use barracuda::linalg::eigh_f64;
///
/// // Symmetric 2×2 matrix: [[3, 1], [1, 3]]
/// // Eigenvalues: 2, 4
/// let a = vec![3.0, 1.0, 1.0, 3.0];
/// let eig = eigh_f64(&a, 2)?;
///
/// assert!((eig.eigenvalues[0] - 2.0).abs() < 1e-10);
/// assert!((eig.eigenvalues[1] - 4.0).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # Notes
///
/// - For best precision, ensure the input matrix is exactly symmetric
/// - The algorithm handles repeated eigenvalues but eigenvectors may be
///   arbitrary within the eigenspace
/// - Complexity: O(n³) per sweep, typically 5-10 sweeps for convergence
pub fn eigh_f64(a: &[f64], n: usize) -> Result<EighDecomposition> {
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

    // Symmetry guard: non-symmetric input produces silently wrong results.
    // Check upper vs lower triangle within machine epsilon tolerance.
    let sym_tol = 1e-10 * a.iter().fold(0.0_f64, |mx, &v| mx.max(v.abs())).max(1e-15);
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (a[i * n + j] - a[j * n + i]).abs();
            if diff > sym_tol {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "Matrix is not symmetric: A[{i},{j}]={} vs A[{j},{i}]={} (diff={diff:.2e})",
                        a[i * n + j],
                        a[j * n + i],
                    ),
                });
            }
        }
    }

    // Special case: 1×1
    if n == 1 {
        return Ok(EighDecomposition {
            eigenvalues: vec![a[0]],
            eigenvectors: vec![1.0],
            n: 1,
        });
    }

    // Copy matrix (will be transformed to diagonal)
    let mut d = a.to_vec();

    // Initialize eigenvectors as identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi iteration parameters
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1e-14;

    for _ in 0..MAX_SWEEPS {
        // Find largest off-diagonal element
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[i * n + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Convergence check
        if max_off < TOL {
            break;
        }

        // Compute rotation angle
        let app = d[p * n + p];
        let aqq = d[q * n + q];
        let apq = d[p * n + q];

        let (c, s) = if (aqq - app).abs() < TOL {
            // θ = π/4
            let c = std::f64::consts::FRAC_1_SQRT_2;
            let s = c * apq.signum();
            (c, s)
        } else {
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                1.0 / (tau - (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;
            (c, s)
        };

        // Apply Givens rotation to D: D' = Jᵀ D J
        // Update rows p and q
        for k in 0..n {
            if k != p && k != q {
                let dkp = d[k * n + p];
                let dkq = d[k * n + q];
                d[k * n + p] = c * dkp - s * dkq;
                d[p * n + k] = d[k * n + p];
                d[k * n + q] = s * dkp + c * dkq;
                d[q * n + k] = d[k * n + q];
            }
        }

        // Update diagonal elements
        let new_pp = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        let new_qq = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        d[p * n + p] = new_pp;
        d[q * n + q] = new_qq;
        d[p * n + q] = 0.0;
        d[q * n + p] = 0.0;

        // Apply rotation to eigenvectors: V' = V J
        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();

    // Sort eigenvalues ascending and reorder eigenvectors
    let mut indexed: Vec<(usize, f64)> = eigenvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_vals: Vec<f64> = indexed.iter().map(|(_, val)| *val).collect();
    let mut sorted_vecs = vec![0.0; n * n];
    for (new_idx, (old_idx, _)) in indexed.iter().enumerate() {
        for row in 0..n {
            sorted_vecs[row * n + new_idx] = v[row * n + old_idx];
        }
    }

    eigenvalues = sorted_vals;

    Ok(EighDecomposition {
        eigenvalues,
        eigenvectors: sorted_vecs,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_eigh_2x2_simple() {
        // [[3, 1], [1, 3]] has eigenvalues 2, 4
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let eig = eigh_f64(&a, 2).unwrap();

        assert!(approx_eq(eig.eigenvalues[0], 2.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 4.0, 1e-10));

        // Verify trace
        assert!(approx_eq(eig.trace(), 6.0, 1e-10));

        // Verify determinant
        assert!(approx_eq(eig.det(), 8.0, 1e-10));
    }

    #[test]
    fn test_eigh_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let eig = eigh_f64(&a, 2).unwrap();

        assert!(approx_eq(eig.eigenvalues[0], 1.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 1.0, 1e-10));
    }

    #[test]
    fn test_eigh_diagonal() {
        // Already diagonal: eigenvalues are diagonal elements
        let a = vec![5.0, 0.0, 0.0, 3.0];
        let eig = eigh_f64(&a, 2).unwrap();

        // Sorted ascending: 3, 5
        assert!(approx_eq(eig.eigenvalues[0], 3.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 5.0, 1e-10));
    }

    #[test]
    fn test_eigh_3x3() {
        // Tridiagonal matrix with known eigenvalues
        // [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Eigenvalues: 2 - √2, 2, 2 + √2
        let a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let eig = eigh_f64(&a, 3).unwrap();

        let expected = [
            2.0 - std::f64::consts::SQRT_2,
            2.0,
            2.0 + std::f64::consts::SQRT_2,
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                approx_eq(eig.eigenvalues[i], exp, 1e-10),
                "eigenvalue[{}] = {}, expected {}",
                i,
                eig.eigenvalues[i],
                exp
            );
        }

        // Verify trace = 6
        assert!(approx_eq(eig.trace(), 6.0, 1e-10));
    }

    #[test]
    fn test_eigh_reconstruction() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let eig = eigh_f64(&a, 2).unwrap();

        let recon = eig.reconstruct();

        for i in 0..4 {
            assert!(
                approx_eq(recon[i], a[i], 1e-10),
                "recon[{}] = {}, a = {}",
                i,
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_eigh_orthogonality() {
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let eig = eigh_f64(&a, 2).unwrap();

        // V^T V should be identity
        let mut vtv = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    vtv[i * 2 + j] += eig.eigenvectors[k * 2 + i] * eig.eigenvectors[k * 2 + j];
                }
            }
        }

        // Check identity
        assert!(approx_eq(vtv[0], 1.0, 1e-10));
        assert!(approx_eq(vtv[1], 0.0, 1e-10));
        assert!(approx_eq(vtv[2], 0.0, 1e-10));
        assert!(approx_eq(vtv[3], 1.0, 1e-10));
    }

    #[test]
    fn test_eigh_sort_descending() {
        let a = vec![1.0, 0.0, 0.0, 5.0];
        let mut eig = eigh_f64(&a, 2).unwrap();

        // Initially ascending: 1, 5
        assert!(approx_eq(eig.eigenvalues[0], 1.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 5.0, 1e-10));

        eig.sort_descending();

        // Now descending: 5, 1
        assert!(approx_eq(eig.eigenvalues[0], 5.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 1.0, 1e-10));
    }

    #[test]
    fn test_eigh_eigenvector() {
        let a = vec![3.0, 1.0, 1.0, 3.0];
        let eig = eigh_f64(&a, 2).unwrap();

        let v0 = eig.eigenvector(0).unwrap();
        let v1 = eig.eigenvector(1).unwrap();

        // v0 and v1 should be orthogonal
        let dot: f64 = v0.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        assert!(approx_eq(dot, 0.0, 1e-10));

        // v0 and v1 should be unit vectors
        let norm0: f64 = v0.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(approx_eq(norm0, 1.0, 1e-10));
        assert!(approx_eq(norm1, 1.0, 1e-10));
    }

    #[test]
    fn test_eigh_1x1() {
        let a = vec![7.0];
        let eig = eigh_f64(&a, 1).unwrap();

        assert_eq!(eig.eigenvalues.len(), 1);
        assert!(approx_eq(eig.eigenvalues[0], 7.0, 1e-10));
    }

    #[test]
    fn test_eigh_negative_eigenvalues() {
        // [[0, 1], [1, 0]] has eigenvalues -1, 1
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let eig = eigh_f64(&a, 2).unwrap();

        assert!(approx_eq(eig.eigenvalues[0], -1.0, 1e-10));
        assert!(approx_eq(eig.eigenvalues[1], 1.0, 1e-10));
    }

    #[test]
    fn test_eigh_errors() {
        assert!(eigh_f64(&[], 0).is_err());
        assert!(eigh_f64(&[1.0, 2.0, 3.0], 2).is_err());
    }
}
