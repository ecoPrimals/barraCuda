// SPDX-License-Identifier: AGPL-3.0-only
//! Non-Negative Matrix Factorization (NMF) — Lee & Seung multiplicative updates.
//!
//! Factorises a non-negative matrix **V** (m x n) into two non-negative factors
//! **W** (m x k) and **H** (k x n) such that V ≈ WH.
//!
//! Two objective functions are supported:
//! - **Euclidean** (Frobenius): minimise ‖V − `WH‖²_F`
//! - **KL divergence**: minimise `D_KL(V` ‖ WH)
//!
//! Absorbed from wetSpring bio module (Feb 2026). CPU-only implementation;
//! GPU WGSL generation is future work (OdeSystem-style trait pattern).
//!
//! # References
//!
//! Lee, D.D. & Seung, H.S. (1999). Learning the parts of objects by
//! non-negative matrix factorization. *Nature*, 401, 788–791.
//!
//! Lee, D.D. & Seung, H.S. (2000). Algorithms for Non-negative Matrix
//! Factorization. *`NeurIPS` 2000*.

use crate::error::BarracudaError;

/// Result of an NMF factorisation.
#[derive(Debug, Clone)]
pub struct NmfResult {
    /// Factor matrix W (m x k), stored row-major.
    pub w: Vec<f64>,
    /// Factor matrix H (k x n), stored row-major.
    pub h: Vec<f64>,
    /// Number of rows in V.
    pub m: usize,
    /// Factorisation rank.
    pub k: usize,
    /// Number of columns in V.
    pub n: usize,
    /// Reconstruction error at each iteration.
    pub errors: Vec<f64>,
}

/// Objective function for NMF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmfObjective {
    /// Minimise ‖V − `WH‖²_F` (Frobenius norm).
    Euclidean,
    /// Minimise `D_KL(V` ‖ WH) (generalised Kullback-Leibler divergence).
    KlDivergence,
}

/// Configuration for NMF.
#[derive(Debug, Clone)]
pub struct NmfConfig {
    /// Number of latent factors.
    pub rank: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative change in error).
    pub tol: f64,
    /// Objective function to minimise.
    pub objective: NmfObjective,
    /// Random seed for initialisation.
    pub seed: u64,
}

impl Default for NmfConfig {
    fn default() -> Self {
        Self {
            rank: 10,
            max_iter: 200,
            tol: 1e-4,
            objective: NmfObjective::Euclidean,
            seed: 42,
        }
    }
}

/// Simple LCG PRNG for reproducible initialisation (no external dep).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let bits = (self.0 >> 11) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }
}

/// Run NMF with Lee & Seung multiplicative update rules.
///
/// `v` is the input matrix (m x n) in row-major order. All entries must be >= 0.
///
/// # Errors
///
/// Returns `BarracudaError::InvalidShape` if dimensions are inconsistent or rank
/// is out of range.
pub fn nmf(v: &[f64], m: usize, n: usize, config: &NmfConfig) -> Result<NmfResult, BarracudaError> {
    let k = config.rank;
    if v.len() != m * n {
        return Err(BarracudaError::InvalidShape {
            expected: vec![m, n],
            actual: vec![v.len()],
        });
    }
    if k == 0 || k > m.min(n) {
        return Err(BarracudaError::InvalidShape {
            expected: vec![1, m.min(n)],
            actual: vec![k],
        });
    }

    let mut rng = Lcg::new(config.seed);

    let mut w = vec![0.0; m * k];
    let mut h = vec![0.0; k * n];
    for val in &mut w {
        *val = rng.next_f64() * 0.1 + 1e-10;
    }
    for val in &mut h {
        *val = rng.next_f64() * 0.1 + 1e-10;
    }

    let mut errors = Vec::with_capacity(config.max_iter);
    let eps = 1e-15;

    match config.objective {
        NmfObjective::Euclidean => {
            nmf_euclidean(v, &mut w, &mut h, m, n, k, config, &mut errors, eps);
        }
        NmfObjective::KlDivergence => {
            nmf_kl(v, &mut w, &mut h, m, n, k, config, &mut errors, eps);
        }
    }

    Ok(NmfResult {
        w,
        h,
        m,
        k,
        n,
        errors,
    })
}

/// Euclidean (Frobenius) NMF — multiplicative updates.
fn nmf_euclidean(
    v: &[f64],
    w: &mut [f64],
    h: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    config: &NmfConfig,
    errors: &mut Vec<f64>,
    eps: f64,
) {
    let mut wh = vec![0.0; m * n];
    let mut v_ht = vec![0.0; m * k];
    let mut w_h_ht = vec![0.0; m * k];
    let mut wt_v = vec![0.0; k * n];
    let mut wt_w_h = vec![0.0; k * n];
    let mut h_ht = vec![0.0; k * k];
    let mut wt_w = vec![0.0; k * k];

    for iter in 0..config.max_iter {
        matmul(w, h, &mut wh, m, k, n);
        let err = frobenius_error(v, &wh);
        errors.push(err);
        if iter > 0 && (errors[iter - 1] - err).abs() / (errors[iter - 1] + eps) < config.tol {
            break;
        }

        matmul_at_b(w, v, &mut wt_v, m, k, n);
        matmul_at_b(w, w, &mut wt_w, m, k, k);
        matmul(&wt_w, h, &mut wt_w_h, k, k, n);
        for i in 0..k * n {
            h[i] *= wt_v[i] / (wt_w_h[i] + eps);
        }

        matmul_a_bt(v, h, &mut v_ht, m, n, k);
        matmul_a_bt(h, h, &mut h_ht, k, n, k);
        matmul(w, &h_ht, &mut w_h_ht, m, k, k);
        for i in 0..m * k {
            w[i] *= v_ht[i] / (w_h_ht[i] + eps);
        }
    }
}

/// KL-divergence NMF — multiplicative updates.
fn nmf_kl(
    v: &[f64],
    w: &mut [f64],
    h: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    config: &NmfConfig,
    errors: &mut Vec<f64>,
    eps: f64,
) {
    let mut wh = vec![0.0; m * n];
    let mut ratio = vec![0.0; m * n];
    let mut wt_ratio = vec![0.0; k * n];
    let mut ratio_ht = vec![0.0; m * k];

    for iter in 0..config.max_iter {
        matmul(w, h, &mut wh, m, k, n);
        let err = kl_divergence(v, &wh, eps);
        errors.push(err);
        if iter > 0 && (errors[iter - 1] - err).abs() / (errors[iter - 1] + eps) < config.tol {
            break;
        }

        for i in 0..m * n {
            ratio[i] = v[i] / (wh[i] + eps);
        }

        matmul_at_b(w, &ratio, &mut wt_ratio, m, k, n);
        let w_col_sums = col_sums_of_transposed(w, m, k);
        for j in 0..n {
            for kk in 0..k {
                h[kk * n + j] *= wt_ratio[kk * n + j] / (w_col_sums[kk] + eps);
            }
        }

        matmul(w, h, &mut wh, m, k, n);
        for i in 0..m * n {
            ratio[i] = v[i] / (wh[i] + eps);
        }

        matmul_a_bt(&ratio, h, &mut ratio_ht, m, n, k);
        let h_row_sums = row_sums(h, k, n);
        for i in 0..m {
            for kk in 0..k {
                w[i * k + kk] *= ratio_ht[i * k + kk] / (h_row_sums[kk] + eps);
            }
        }
    }
}

// ── Dense linear algebra helpers ─────────────────────────────────

fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}

fn matmul_at_b(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[p * n + j] += a_ip * b[i * n + j];
            }
        }
    }
}

fn matmul_a_bt(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..n {
            let a_ip = a[i * n + p];
            for j in 0..k {
                c[i * k + j] += a_ip * b[j * n + p];
            }
        }
    }
}

fn frobenius_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn kl_divergence(v: &[f64], wh: &[f64], eps: f64) -> f64 {
    v.iter()
        .zip(wh.iter())
        .map(|(vi, whi)| {
            if *vi > eps {
                vi * (vi / (whi + eps)).ln() - vi + whi
            } else {
                *whi
            }
        })
        .sum()
}

fn col_sums_of_transposed(a: &[f64], m: usize, k: usize) -> Vec<f64> {
    let mut sums = vec![0.0; k];
    for i in 0..m {
        for j in 0..k {
            sums[j] += a[i * k + j];
        }
    }
    sums
}

fn row_sums(a: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mut sums = vec![0.0; k];
    for i in 0..k {
        for j in 0..n {
            sums[i] += a[i * n + j];
        }
    }
    sums
}

// ── Scoring helpers ──────────────────────────────────────────────

/// Cosine similarity between two vectors.
/// # Panics
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Score all row-column pairs by reconstructing V ≈ WH and return
/// the top-K pairs sorted by descending score.
///
/// Returns `Vec<(row_idx, col_idx, score)>`.
/// # Panics
/// Panics if `partial_cmp` returns `None` during sort (e.g. NaN values).
#[must_use]
pub fn top_k_predictions(result: &NmfResult, top_k: usize) -> Vec<(usize, usize, f64)> {
    let m = result.m;
    let n = result.n;
    let k = result.k;

    let mut v_hat = vec![0.0; m * n];
    matmul(&result.w, &result.h, &mut v_hat, m, k, n);

    let mut pairs: Vec<(usize, usize, f64)> = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            pairs.push((i, j, v_hat[i * n + j]));
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);
    pairs
}

/// Reconstruction error: ‖V − WH‖_F / ‖V‖_F (relative).
#[must_use]
pub fn relative_reconstruction_error(v: &[f64], result: &NmfResult) -> f64 {
    let mut wh = vec![0.0; result.m * result.n];
    matmul(&result.w, &result.h, &mut wh, result.m, result.k, result.n);
    let err = frobenius_error(v, &wh);
    let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if v_norm < 1e-15 {
        return 0.0;
    }
    err / v_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_converges() {
        let v = vec![
            1.0, 2.0, 0.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0,
            0.0, 3.0, 1.0,
        ];
        let config = NmfConfig {
            rank: 2,
            max_iter: 100,
            tol: 1e-6,
            objective: NmfObjective::Euclidean,
            seed: 42,
        };
        let result = nmf(&v, 4, 5, &config).expect("nmf");
        assert_eq!(result.m, 4);
        assert_eq!(result.n, 5);
        assert_eq!(result.k, 2);
        assert!(!result.errors.is_empty());
        for pair in result.errors.windows(2) {
            assert!(
                pair[1] <= pair[0] + 1e-10,
                "error not monotonically decreasing"
            );
        }
        let rel_err = relative_reconstruction_error(&v, &result);
        assert!(rel_err < 0.5, "relative error {rel_err} too high");
    }

    #[test]
    fn kl_divergence_converges() {
        let v = vec![
            1.0, 2.0, 0.5, 1.0, 0.5, 0.1, 1.0, 3.0, 0.1, 2.0, 2.0, 0.1, 1.0, 2.0, 0.1, 1.0, 1.0,
            0.1, 3.0, 1.0,
        ];
        let config = NmfConfig {
            rank: 2,
            max_iter: 100,
            tol: 1e-6,
            objective: NmfObjective::KlDivergence,
            seed: 123,
        };
        let result = nmf(&v, 4, 5, &config).expect("nmf");
        assert!(!result.errors.is_empty());
        let last = *result.errors.last().unwrap();
        assert!(last < result.errors[0], "KL error should decrease");
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn cosine_45_degrees() {
        let a = [1.0, 0.0, 0.0];
        let c = [1.0, 1.0, 0.0];
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((cosine_similarity(&a, &c) - expected).abs() < 1e-10);
    }

    #[test]
    fn top_k_diagonal_dominant() {
        let v = vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0];
        let config = NmfConfig {
            rank: 2,
            max_iter: 200,
            tol: 1e-8,
            objective: NmfObjective::Euclidean,
            seed: 42,
        };
        let result = nmf(&v, 3, 3, &config).expect("nmf");
        let top = top_k_predictions(&result, 3);
        assert_eq!(top.len(), 3);
        assert!(top[0].2 > 1.0);
    }

    #[test]
    fn invalid_shape_rejected() {
        let v = vec![1.0, 2.0, 3.0];
        let config = NmfConfig {
            rank: 2,
            ..NmfConfig::default()
        };
        assert!(nmf(&v, 4, 5, &config).is_err());
    }

    #[test]
    fn zero_rank_rejected() {
        let v = vec![1.0; 12];
        let config = NmfConfig {
            rank: 0,
            ..NmfConfig::default()
        };
        assert!(nmf(&v, 3, 4, &config).is_err());
    }

    #[test]
    fn factors_are_nonnegative() {
        let v = vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 1.0, 2.0];
        let config = NmfConfig {
            rank: 2,
            max_iter: 50,
            ..NmfConfig::default()
        };
        let result = nmf(&v, 3, 4, &config).expect("nmf");
        assert!(result.w.iter().all(|x| *x >= 0.0), "W must be non-negative");
        assert!(result.h.iter().all(|x| *x >= 0.0), "H must be non-negative");
    }
}
