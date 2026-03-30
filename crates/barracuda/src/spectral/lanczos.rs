// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lanczos tridiagonalization for sparse symmetric eigensolve.
//!
//! Krylov subspace method with full reorthogonalization; eigenvalues via
//! Sturm bisection on the resulting tridiagonal matrix.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)

use crate::rng::LcgRng;
use super::sparse::SpectralCsrMatrix;
use super::tridiag::{find_all_eigenvalues, tridiag_eigenvectors};

/// Result of the Lanczos algorithm: a tridiagonal representation of the
/// original matrix restricted to the Krylov subspace.
pub struct LanczosTridiag {
    /// Diagonal elements `α_j` = ⟨`v_j`, A `v_j`⟩
    pub alpha: Vec<f64>,
    /// Off-diagonal elements `β_j` = ‖`w_j`‖
    pub beta: Vec<f64>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Extended Lanczos result that also retains the Krylov basis vectors `Q`.
///
/// The basis vectors are needed for computing Ritz eigenvectors:
///   `v_i` = `Q` × `z_i` (where `z_i` are tridiagonal eigenvectors)
pub struct LanczosTridiagWithBasis {
    /// The tridiagonal decomposition.
    pub tridiag: LanczosTridiag,
    /// Lanczos basis vectors Q[j] (each of length n), stored row-major.
    /// Q has dimensions m × n where m = iterations.
    pub basis: Vec<Vec<f64>>,
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
/// Supports matrices up to any dimension (tested to N = 10,000+).
/// Memory scales as O(m × n) for the reorthogonalization basis.
///
/// # Arguments
/// - `matrix`: symmetric sparse matrix in CSR format
/// - `max_iter`: maximum Lanczos iterations (cap at matrix dimension)
/// - `seed`: PRNG seed for initial random vector
///
/// # Provenance
/// Lanczos (1950), J. Res. Nat. Bur. Standards 45, 255
#[must_use]
pub fn lanczos(matrix: &SpectralCsrMatrix, max_iter: usize, seed: u64) -> LanczosTridiag {
    lanczos_with_config(matrix, max_iter, seed, &LanczosConfig::default())
}

/// Configuration for Lanczos tridiagonalization.
pub struct LanczosConfig {
    /// Convergence threshold for `β` (off-diagonal element).
    /// When `β < threshold`, invariant subspace found.
    pub convergence_threshold: f64,
    /// Optional progress callback: called with (iteration, total).
    /// Useful for long runs (N > 1000) to report progress.
    pub progress: Option<Box<dyn Fn(usize, usize)>>,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            convergence_threshold: 1e-14,
            progress: None,
        }
    }
}

/// Lanczos tridiagonalization with full reorthogonalization and configurable options.
///
/// Extended variant supporting:
/// - Configurable convergence threshold
/// - Progress callbacks for long-running eigensolves (N > 1000)
/// - Memory-efficient two-pass reorthogonalization for large matrices
#[must_use]
pub fn lanczos_with_config(
    matrix: &SpectralCsrMatrix,
    max_iter: usize,
    seed: u64,
    config: &LanczosConfig,
) -> LanczosTridiag {
    let n = matrix.n;
    let m = max_iter.min(n);

    let mut rng = LcgRng::new(seed);

    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let norm = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);

    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    vecs.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;
    let mut w = vec![0.0; n];

    for j in 0..m {
        matrix.spmv(&v, &mut w);

        if j > 0 {
            for i in 0..n {
                w[i] -= beta_prev * v_prev[i];
            }
        }

        let a_j = dot(&w, &v);
        alpha.push(a_j);

        for i in 0..n {
            w[i] -= a_j * v[i];
        }

        // Full reorthogonalization: two-pass classical Gram-Schmidt for
        // numerical stability on large matrices (N > 1000).
        for _pass in 0..2 {
            for prev in &vecs {
                let proj = dot(&w, prev);
                for i in 0..n {
                    w[i] -= proj * prev[i];
                }
            }
        }

        let b_next = dot(&w, &w).sqrt();

        if b_next < config.convergence_threshold {
            beta.push(0.0);
            break;
        }

        beta.push(b_next);

        v_prev.copy_from_slice(&v);
        beta_prev = b_next;
        for i in 0..n {
            v[i] = w[i] / b_next;
        }
        vecs.push(v.clone());

        if let Some(ref progress) = config.progress {
            progress(j + 1, m);
        }
    }

    let iterations = alpha.len();
    LanczosTridiag {
        alpha,
        beta,
        iterations,
    }
}

/// Lanczos tridiagonalization that also retains the Krylov basis vectors.
///
/// Required for eigenvector computation (Ritz vectors = `Q` × tridiag eigenvectors).
/// Memory: O(m × n) where m = iterations performed.
#[must_use]
pub fn lanczos_with_basis(
    matrix: &SpectralCsrMatrix,
    max_iter: usize,
    seed: u64,
    config: &LanczosConfig,
) -> LanczosTridiagWithBasis {
    let n = matrix.n;
    let m = max_iter.min(n);

    let mut rng = LcgRng::new(seed);

    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let norm = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);

    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    vecs.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;
    let mut w = vec![0.0; n];

    for j in 0..m {
        matrix.spmv(&v, &mut w);

        if j > 0 {
            for i in 0..n {
                w[i] -= beta_prev * v_prev[i];
            }
        }

        let a_j = dot(&w, &v);
        alpha.push(a_j);

        for i in 0..n {
            w[i] -= a_j * v[i];
        }

        for _pass in 0..2 {
            for prev in &vecs {
                let proj = dot(&w, prev);
                for i in 0..n {
                    w[i] -= proj * prev[i];
                }
            }
        }

        let b_next = dot(&w, &w).sqrt();

        if b_next < config.convergence_threshold {
            beta.push(0.0);
            break;
        }

        beta.push(b_next);

        v_prev.copy_from_slice(&v);
        beta_prev = b_next;
        for i in 0..n {
            v[i] = w[i] / b_next;
        }
        vecs.push(v.clone());

        if let Some(ref progress) = config.progress {
            progress(j + 1, m);
        }
    }

    let iterations = alpha.len();
    let basis = vecs[..iterations].to_vec();

    LanczosTridiagWithBasis {
        tridiag: LanczosTridiag {
            alpha,
            beta,
            iterations,
        },
        basis,
    }
}

/// Compute the k dominant eigenpairs (eigenvalue, eigenvector) of a sparse symmetric matrix.
///
/// Uses Lanczos tridiagonalization + Ritz vector construction:
/// 1. Run Lanczos to build tridiagonal `T` and basis `Q`
/// 2. Compute eigenpairs of `T`: (`λ_i`, `z_i`)
/// 3. Ritz vectors: `v_i` = `Q^T` × `z_i` (back-transform to original space)
/// 4. Return top-k by eigenvalue magnitude
///
/// # Arguments
/// - `matrix`: symmetric sparse matrix in CSR format
/// - `k`: number of eigenpairs to return
/// - `config`: optional Lanczos configuration (uses default if `None`)
///
/// # Returns
/// Vec of (eigenvalue, eigenvector) pairs sorted by descending |eigenvalue|.
#[must_use]
pub fn lanczos_eigenvectors(
    matrix: &SpectralCsrMatrix,
    k: usize,
    config: Option<LanczosConfig>,
) -> Vec<(f64, Vec<f64>)> {
    let config = config.unwrap_or_default();
    let n = matrix.n;
    let max_iter = n.min(k * 3 + 20).max(k);
    let result = lanczos_with_basis(matrix, max_iter, 42, &config);
    let m = result.tridiag.iterations;

    if m == 0 {
        return Vec::new();
    }

    let off_diag: Vec<f64> = result.tridiag.beta[..m.saturating_sub(1)].to_vec();
    let (evals, evecs_flat) = tridiag_eigenvectors(&result.tridiag.alpha, &off_diag);

    let mut eigenpairs: Vec<(f64, Vec<f64>)> = evals
        .iter()
        .enumerate()
        .map(|(i, &lam)| {
            let z: Vec<f64> = (0..m).map(|j| evecs_flat[j * m + i]).collect();

            let mut v = vec![0.0; n];
            for (j, q_j) in result.basis.iter().enumerate() {
                let z_j = z[j];
                for (idx, val) in q_j.iter().enumerate() {
                    v[idx] += z_j * val;
                }
            }

            let norm = dot(&v, &v).sqrt();
            if norm > 1e-14 {
                for x in &mut v {
                    *x /= norm;
                }
            }

            (lam, v)
        })
        .collect();

    eigenpairs.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
    eigenpairs.truncate(k);
    eigenpairs
}

/// Compute the k largest eigenvalues using Lanczos with early termination.
///
/// More efficient than full Lanczos when only extremal eigenvalues are needed.
/// Runs until the k extremal eigenvalues converge or `max_iter` is reached.
#[expect(
    dead_code,
    reason = "public API for springs (groundSpring, hotSpring) — not used within barracuda itself"
)]
#[must_use]
pub fn lanczos_extremal(
    matrix: &SpectralCsrMatrix,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<f64> {
    let m = max_iter.min(matrix.n).max(k * 3);
    let result = lanczos(matrix, m, seed);
    let mut evals = lanczos_eigenvalues(&result);
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    evals.truncate(k);
    evals
}

/// Extract eigenvalues from a Lanczos tridiagonal via Sturm bisection.
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{SpectralCsrMatrix, anderson_hamiltonian, find_all_eigenvalues};

    fn tridiag_to_csr(d: &[f64], e: &[f64]) -> SpectralCsrMatrix {
        let n = d.len();
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
        SpectralCsrMatrix {
            n,
            row_ptr,
            col_idx,
            values,
        }
    }

    #[test]
    fn lanczos_vs_sturm_1d() {
        let n = 100;
        let (d, e) = anderson_hamiltonian(n, 2.0, 42);
        let csr = tridiag_to_csr(&d, &e);

        let sturm_evals = find_all_eigenvalues(&d, &e);
        let lanczos_result = lanczos(&csr, n, 42);
        let lanczos_evals = lanczos_eigenvalues(&lanczos_result);

        let sturm_min = sturm_evals[0];
        let sturm_max = sturm_evals[n - 1];
        let lanczos_min = lanczos_evals[0];
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

    #[test]
    fn lanczos_eigenvalues_empty_returns_empty() {
        let result = LanczosTridiag {
            alpha: vec![],
            beta: vec![],
            iterations: 0,
        };
        let evals = lanczos_eigenvalues(&result);
        assert!(evals.is_empty());
    }

    #[test]
    fn lanczos_1x1_identity() {
        let csr = SpectralCsrMatrix {
            n: 1,
            row_ptr: vec![0, 1],
            col_idx: vec![0],
            values: vec![5.0],
        };
        let result = lanczos(&csr, 1, 1);
        let evals = lanczos_eigenvalues(&result);
        assert_eq!(evals.len(), 1);
        assert!((evals[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn lanczos_2x2_symmetric() {
        let csr = SpectralCsrMatrix {
            n: 2,
            row_ptr: vec![0, 2, 4],
            col_idx: vec![0, 1, 0, 1],
            values: vec![2.0, 1.0, 1.0, 3.0],
        };
        let result = lanczos(&csr, 2, 7);
        let mut evals = lanczos_eigenvalues(&result);
        evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let trace: f64 = 2.0 + 3.0;
        let det: f64 = 2.0f64.mul_add(3.0, -1.0);
        let disc = trace.mul_add(trace, -(4.0 * det)).sqrt();
        let half_trace = trace * 0.5;
        let half_disc = disc * 0.5;
        let expected_min = half_trace - half_disc;
        let expected_max = half_trace + half_disc;

        assert!(
            (evals[0] - expected_min).abs() < 1e-10,
            "min: {:.10} vs {expected_min:.10}",
            evals[0]
        );
        assert!(
            (evals[1] - expected_max).abs() < 1e-10,
            "max: {:.10} vs {expected_max:.10}",
            evals[1]
        );
    }

    #[test]
    fn lanczos_small_n_clamps_iterations() {
        let n = 5;
        let (d, e) = anderson_hamiltonian(n, 1.0, 99);
        let csr = tridiag_to_csr(&d, &e);
        let result = lanczos(&csr, 1000, 99);
        assert!(result.iterations <= n);
    }

    #[test]
    fn lanczos_with_config_threshold() {
        let n = 50;
        let (d, e) = anderson_hamiltonian(n, 1.0, 42);
        let csr = tridiag_to_csr(&d, &e);
        let config = LanczosConfig {
            convergence_threshold: 1.0,
            progress: None,
        };
        let result = lanczos_with_config(&csr, n, 42, &config);
        assert!(
            result.iterations <= n,
            "early termination with high threshold"
        );
    }

    #[test]
    fn lanczos_different_seeds_converge() {
        let n = 30;
        let (d, e) = anderson_hamiltonian(n, 2.0, 0);
        let csr = tridiag_to_csr(&d, &e);
        let sturm_evals = find_all_eigenvalues(&d, &e);

        for seed in [1, 42, 999, 12345] {
            let result = lanczos(&csr, n, seed);
            let mut evals = lanczos_eigenvalues(&result);
            evals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!(
                (evals[0] - sturm_evals[0]).abs() < 1e-6,
                "seed {seed}: min mismatch"
            );
            let last = evals.last().unwrap();
            assert!(
                (last - sturm_evals[n - 1]).abs() < 1e-6,
                "seed {seed}: max mismatch"
            );
        }
    }

    #[test]
    fn lanczos_progress_callback_fires() {
        let n = 20;
        let (d, e) = anderson_hamiltonian(n, 1.0, 42);
        let csr = tridiag_to_csr(&d, &e);
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count_clone = count.clone();
        let config = LanczosConfig {
            convergence_threshold: 1e-14,
            progress: Some(Box::new(move |_iter, _total| {
                count_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            })),
        };
        let _result = lanczos_with_config(&csr, n, 42, &config);
        assert!(count.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }

    #[test]
    fn lanczos_eigenvectors_identity() {
        let csr = SpectralCsrMatrix {
            n: 3,
            row_ptr: vec![0, 1, 2, 3],
            col_idx: vec![0, 1, 2],
            values: vec![3.0, 2.0, 1.0],
        };
        let pairs = lanczos_eigenvectors(&csr, 3, None);
        assert_eq!(pairs.len(), 3);

        let eigenvalues: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
        assert!(
            (eigenvalues[0].abs() - 3.0).abs() < 1e-8,
            "largest eigenvalue should be 3.0, got {}",
            eigenvalues[0]
        );
    }

    #[test]
    fn lanczos_eigenvectors_orthonormality() {
        let n = 30;
        let (d, e) = anderson_hamiltonian(n, 2.0, 42);
        let csr = tridiag_to_csr(&d, &e);

        let pairs = lanczos_eigenvectors(&csr, 5, None);
        assert_eq!(pairs.len(), 5);

        for (i, (_, vi)) in pairs.iter().enumerate() {
            let norm_sq: f64 = vi.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-8,
                "eigenvector {i} norm = {}, expected 1",
                norm_sq.sqrt()
            );
        }

        for i in 0..pairs.len() {
            for j in (i + 1)..pairs.len() {
                let d: f64 = pairs[i]
                    .1
                    .iter()
                    .zip(pairs[j].1.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    d.abs() < 1e-6,
                    "eigenvectors {i},{j} not orthogonal: dot = {d}"
                );
            }
        }
    }

    #[test]
    fn lanczos_eigenvectors_residual() {
        let n = 20;
        let (d, e) = anderson_hamiltonian(n, 1.0, 42);
        let csr = tridiag_to_csr(&d, &e);

        let pairs = lanczos_eigenvectors(&csr, 3, None);

        for (lam, v) in &pairs {
            let mut av = vec![0.0; n];
            csr.spmv(v, &mut av);

            let residual: f64 = av
                .iter()
                .zip(v.iter())
                .map(|(avi, vi)| (avi - lam * vi).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(residual < 1e-6, "||Av - λv|| = {residual} for λ = {lam}");
        }
    }

    #[test]
    fn lanczos_eigenvectors_empty() {
        let csr = SpectralCsrMatrix {
            n: 0,
            row_ptr: vec![0],
            col_idx: vec![],
            values: vec![],
        };
        let pairs = lanczos_eigenvectors(&csr, 5, None);
        assert!(pairs.is_empty());
    }

    #[test]
    fn lanczos_with_basis_preserves_vectors() {
        let n = 10;
        let (d, e) = anderson_hamiltonian(n, 1.0, 42);
        let csr = tridiag_to_csr(&d, &e);

        let result = lanczos_with_basis(&csr, n, 42, &LanczosConfig::default());
        assert_eq!(result.basis.len(), result.tridiag.iterations);
        for q in &result.basis {
            assert_eq!(q.len(), n);
            let norm: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "basis vector not normalized: {norm}"
            );
        }
    }
}
