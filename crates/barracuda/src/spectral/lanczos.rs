// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lanczos tridiagonalization for sparse symmetric eigensolve.
//!
//! Krylov subspace method with full reorthogonalization; eigenvalues via
//! Sturm bisection on the resulting tridiagonal matrix.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)

use super::anderson::LcgRng;
use super::sparse::SpectralCsrMatrix;
use super::tridiag::find_all_eigenvalues;

/// Result of the Lanczos algorithm: a tridiagonal representation of the
/// original matrix restricted to the Krylov subspace.
pub struct LanczosTridiag {
    /// Diagonal elements α_j = ⟨v_j, A v_j⟩
    pub alpha: Vec<f64>,
    /// Off-diagonal elements β_j = ‖w_j‖
    pub beta: Vec<f64>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Lanczos tridiagonalization with full reorthogonalization.
///
/// Builds an m-step Krylov subspace for the sparse symmetric matrix A.
/// The eigenvalues of the resulting tridiagonal matrix approximate the
/// eigenvalues of A, with extremal eigenvalues converging first.
///
/// With full reorthogonalization and m = n, the tridiagonal eigenvalues
/// are the exact eigenvalues of A (up to machine precision).
///
/// # Arguments
/// - `matrix`: symmetric sparse matrix in CSR format
/// - `max_iter`: maximum Lanczos iterations (cap at matrix dimension)
/// - `seed`: PRNG seed for initial random vector
///
/// # Provenance
/// Lanczos (1950), J. Res. Nat. Bur. Standards 45, 255
pub fn lanczos(matrix: &SpectralCsrMatrix, max_iter: usize, seed: u64) -> LanczosTridiag {
    let n = matrix.n;
    let m = max_iter.min(n);

    let mut rng = LcgRng::new(seed);

    // Random starting vector, normalized
    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let norm = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);

    // Store all Lanczos vectors for reorthogonalization
    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    vecs.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;
    let mut w = vec![0.0; n];

    for j in 0..m {
        // w = A * v_j
        matrix.spmv(&v, &mut w);

        // w = w - β_j * v_{j-1}
        if j > 0 {
            for i in 0..n {
                w[i] -= beta_prev * v_prev[i];
            }
        }

        // α_j = ⟨w, v_j⟩
        let a_j = dot(&w, &v);
        alpha.push(a_j);

        // w = w - α_j * v_j
        for i in 0..n {
            w[i] -= a_j * v[i];
        }

        // Full reorthogonalization (Gram-Schmidt against all previous vectors)
        for prev in &vecs {
            let proj = dot(&w, prev);
            for i in 0..n {
                w[i] -= proj * prev[i];
            }
        }

        // β_{j+1} = ‖w‖
        let b_next = dot(&w, &w).sqrt();

        if b_next < 1e-14 {
            // Invariant subspace found — Lanczos has converged
            beta.push(0.0);
            break;
        }

        beta.push(b_next);

        // v_{j+1} = w / β_{j+1}
        v_prev.copy_from_slice(&v);
        beta_prev = b_next;
        for i in 0..n {
            v[i] = w[i] / b_next;
        }
        vecs.push(v.clone());
    }

    LanczosTridiag {
        iterations: alpha.len(),
        alpha,
        beta,
    }
}

/// Extract eigenvalues from a Lanczos tridiagonal via Sturm bisection.
pub fn lanczos_eigenvalues(result: &LanczosTridiag) -> Vec<f64> {
    let m = result.iterations;
    if m == 0 {
        return Vec::new();
    }

    let off_diag: Vec<f64> = result.beta[..m.saturating_sub(1)].to_vec();
    find_all_eigenvalues(&result.alpha, &off_diag)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{anderson_hamiltonian, find_all_eigenvalues, SpectralCsrMatrix};

    #[test]
    fn lanczos_vs_sturm_1d() {
        let n = 100;
        let (d, e) = anderson_hamiltonian(n, 2.0, 42);

        // Build 1D Anderson as CSR for Lanczos
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(e[i - 1]);
            }
            col_idx.push(i);
            values.push(d[i]);
            if i < n - 1 {
                col_idx.push(i + 1);
                values.push(e[i]);
            }
            row_ptr.push(col_idx.len());
        }
        let csr = SpectralCsrMatrix {
            n,
            row_ptr,
            col_idx,
            values,
        };

        let sturm_evals = find_all_eigenvalues(&d, &e);
        let lanczos_result = lanczos(&csr, n, 42);
        let lanczos_evals = lanczos_eigenvalues(&lanczos_result);

        let sturm_min = sturm_evals[0];
        let sturm_max = sturm_evals[n - 1];
        let lanczos_min = lanczos_evals[0];
        // infallible: lanczos(&csr, n, seed) with n=100 yields m=100 iterations, so lanczos_evals is non-empty
        let lanczos_max = *lanczos_evals.last().expect("collection verified non-empty");

        assert!(
            (sturm_min - lanczos_min).abs() < 1e-8,
            "min: Sturm={sturm_min:.8}, Lanczos={lanczos_min:.8}"
        );
        assert!(
            (sturm_max - lanczos_max).abs() < 1e-8,
            "max: Sturm={sturm_max:.8}, Lanczos={lanczos_max:.8}"
        );
    }
}
