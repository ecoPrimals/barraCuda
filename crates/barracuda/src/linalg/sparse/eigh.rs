// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sparse symmetric eigenvalue solver via Lanczos tridiagonalization.
//!
//! Bridges `linalg::sparse::CsrMatrix` to the `spectral::lanczos` pipeline.
//! For symmetric n×n sparse matrices, computes eigenvalues (and optionally
//! the k smallest/largest) using the Krylov subspace method.
//!
//! Provenance: hotSpring HFB nuclear structure → toadStool evolution

use crate::error::{BarracudaError, Result};
use crate::spectral::lanczos::{lanczos, lanczos_eigenvalues};
use crate::spectral::sparse::SpectralCsrMatrix;

use super::CsrMatrix;

/// Convert a symmetric `CsrMatrix` to `SpectralCsrMatrix` for Lanczos.
fn to_spectral(matrix: &CsrMatrix) -> Result<SpectralCsrMatrix> {
    if matrix.n_rows != matrix.n_cols {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "sparse_eigh requires square matrix, got {}×{}",
                matrix.n_rows, matrix.n_cols
            ),
        });
    }
    Ok(SpectralCsrMatrix {
        n: matrix.n_rows,
        row_ptr: matrix.row_ptr.clone(),
        col_idx: matrix.col_indices.clone(),
        values: matrix.values.clone(),
    })
}

/// Result of a sparse eigenvalue computation.
#[derive(Debug, Clone)]
pub struct SparseEighResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Compute eigenvalues of a sparse symmetric matrix via Lanczos.
///
/// # Arguments
/// - `matrix`: symmetric sparse matrix (CSR). Only the stored triangle is
///   used; the caller must ensure symmetry.
/// - `max_iter`: maximum Lanczos iterations. For exact eigenvalues, set to `n`.
///   For the k extremal eigenvalues, `2k` to `3k` is usually sufficient.
/// - `seed`: PRNG seed for the initial random vector.
///
/// # Returns
/// Eigenvalues in ascending order. The number of converged eigenvalues
/// depends on `max_iter`: extremal eigenvalues converge first.
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square.
pub fn sparse_eigh(matrix: &CsrMatrix, max_iter: usize, seed: u64) -> Result<SparseEighResult> {
    let spectral = to_spectral(matrix)?;
    let tridiag = lanczos(&spectral, max_iter, seed);
    let eigenvalues = lanczos_eigenvalues(&tridiag);

    Ok(SparseEighResult {
        eigenvalues,
        iterations: tridiag.iterations,
    })
}

/// Compute the k smallest eigenvalues of a sparse symmetric matrix.
///
/// Runs Lanczos with `3*k` iterations (heuristic for convergence of
/// the k lowest Ritz values), then truncates.
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square.
pub fn sparse_eigh_smallest(matrix: &CsrMatrix, k: usize, seed: u64) -> Result<SparseEighResult> {
    let max_iter = (3 * k).min(matrix.n_rows);
    let mut result = sparse_eigh(matrix, max_iter, seed)?;
    result.eigenvalues.truncate(k);
    Ok(result)
}

/// Compute the k largest eigenvalues of a sparse symmetric matrix.
///
/// Runs Lanczos with `3*k` iterations, then takes the last k values.
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square.
pub fn sparse_eigh_largest(matrix: &CsrMatrix, k: usize, seed: u64) -> Result<SparseEighResult> {
    let max_iter = (3 * k).min(matrix.n_rows);
    let mut result = sparse_eigh(matrix, max_iter, seed)?;
    let n = result.eigenvalues.len();
    if n > k {
        result.eigenvalues = result.eigenvalues[n - k..].to_vec();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_eigh_identity() {
        // 3×3 identity → Lanczos converges in 1 step for a scalar multiple.
        // The single Ritz value should be exactly 1.0.
        let matrix = CsrMatrix {
            n_rows: 3,
            n_cols: 3,
            values: vec![1.0, 1.0, 1.0],
            col_indices: vec![0, 1, 2],
            row_ptr: vec![0, 1, 2, 3],
        };
        let result = sparse_eigh(&matrix, 3, 42).unwrap();
        assert!(!result.eigenvalues.is_empty());
        for &ev in &result.eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10, "expected 1.0, got {ev}");
        }
    }

    #[test]
    fn test_sparse_eigh_diagonal() {
        // diag(1, 3, 5) → eigenvalues {1, 3, 5}
        let matrix = CsrMatrix {
            n_rows: 3,
            n_cols: 3,
            values: vec![1.0, 3.0, 5.0],
            col_indices: vec![0, 1, 2],
            row_ptr: vec![0, 1, 2, 3],
        };
        let result = sparse_eigh(&matrix, 3, 123).unwrap();
        assert_eq!(result.eigenvalues.len(), 3);
        let expected = [1.0, 3.0, 5.0];
        for (ev, &exp) in result.eigenvalues.iter().zip(&expected) {
            assert!((ev - exp).abs() < 1e-8, "expected {exp}, got {ev}");
        }
    }

    #[test]
    fn test_sparse_eigh_smallest_k() {
        let matrix = CsrMatrix {
            n_rows: 3,
            n_cols: 3,
            values: vec![1.0, 3.0, 5.0],
            col_indices: vec![0, 1, 2],
            row_ptr: vec![0, 1, 2, 3],
        };
        let result = sparse_eigh_smallest(&matrix, 2, 7).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!((result.eigenvalues[0] - 1.0).abs() < 1e-8);
        assert!((result.eigenvalues[1] - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_non_square_rejected() {
        let matrix = CsrMatrix {
            n_rows: 2,
            n_cols: 3,
            values: vec![1.0, 2.0],
            col_indices: vec![0, 1],
            row_ptr: vec![0, 1, 2],
        };
        assert!(sparse_eigh(&matrix, 2, 0).is_err());
    }
}
