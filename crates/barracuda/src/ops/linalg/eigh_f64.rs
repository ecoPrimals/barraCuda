// SPDX-License-Identifier: AGPL-3.0-only
//! CPU f64 Symmetric Eigenvalue Decomposition: Householder + implicit QR.
//!
//! Achieves LAPACK-level accuracy (~1e-14) at all matrix sizes, replacing
//! the Jacobi iteration for f64 workloads.
//!
//! ## Algorithm
//!
//! 1. **Householder tridiagonalization**: A = Q T Qᵀ in O(4n³/3) flops.
//! 2. **Implicit QL iteration** with Wilkinson shift on T: O(n²) per
//!    iteration, quadratic convergence (2–3 iterations per eigenvalue).
//! 3. **Back-transform**: eigenvectors of T → eigenvectors of A via Q.
//!
//! ## Provenance
//!
//! Absorbed from neuralSpring `src/eigh.rs` (Feb 21, 2026). Validated
//! against LAPACK DSTEQR with 9/9 PASS, machine-epsilon accuracy.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use barracuda::ops::linalg::eigh_f64::{eigh_householder_qr, EighDecompositionF64};
//! let a = vec![3.0, 1.0, 1.0, 3.0]; // 2x2 symmetric
//! let result = eigh_householder_qr(&a, 2);
//! assert!(result.reconstruction_error(&a) < 1e-12);
//! ```

/// NAK-optimized batched symmetric eigensolve (f64).
pub const WGSL_BATCHED_EIGH_NAK_OPTIMIZED: &str =
    include_str!("../../shaders/linalg/batched_eigh_nak_optimized_f64.wgsl");

/// Batched symmetric eigensolve (f64).
pub const WGSL_BATCHED_EIGH_F64: &str = include_str!("../../shaders/linalg/batched_eigh_f64.wgsl");

/// Batched symmetric eigensolve single-dispatch variant (f64).
pub const WGSL_BATCHED_EIGH_SINGLE_DISPATCH_F64: &str =
    include_str!("../../shaders/linalg/batched_eigh_single_dispatch_f64.wgsl");

/// Symmetric eigenvalue decomposition (f64).
pub const WGSL_EIGH_F64: &str = include_str!("../../shaders/linalg/eigh_f64.wgsl");

/// Result of symmetric eigenvalue decomposition A = V·D·Vᵀ (f64).
///
/// Eigenvalues in ascending order; eigenvectors as columns of V (row-major).
#[derive(Debug, Clone)]
pub struct EighDecompositionF64 {
    /// Eigenvalues in ascending order
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of V (n×n, row-major).
    /// Column j is the eigenvector for `eigenvalues[j]`.
    pub eigenvectors: Vec<f64>,
    /// Matrix dimension (n×n)
    pub n: usize,
}

impl EighDecompositionF64 {
    /// Reconstruct A = V·D·Vᵀ for verification.
    #[must_use]
    pub fn reconstruct(&self) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.eigenvectors[i * n + k]
                        * self.eigenvalues[k]
                        * self.eigenvectors[j * n + k];
                }
                a[i * n + j] = s;
            }
        }
        a
    }

    /// Frobenius norm of (A - reconstruct).
    #[must_use]
    pub fn reconstruction_error(&self, a: &[f64]) -> f64 {
        let r = self.reconstruct();
        r.iter()
            .zip(a.iter())
            .map(|(ri, ai)| (ri - ai).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Symmetric eigenvalue decomposition via Householder + implicit QR (f64).
///
/// Input: `a` is an n×n symmetric matrix in row-major order.
/// Returns eigenvalues in ascending order with orthonormal eigenvectors.
///
/// # Panics
///
/// Panics if `a.len() != n * n` or `n == 0`.
#[must_use]
pub fn eigh_householder_qr(a: &[f64], n: usize) -> EighDecompositionF64 {
    assert!(n > 0 && a.len() == n * n, "invalid matrix dimensions");

    if n == 1 {
        return EighDecompositionF64 {
            eigenvalues: vec![a[0]],
            eigenvectors: vec![1.0],
            n: 1,
        };
    }

    if n == 2 {
        return eigh_2x2(a);
    }

    let (diag, off_diag, q) = householder_tridiag(a, n);
    let (eigenvalues, z) = ql_implicit(&diag, &off_diag, n);

    // Back-transform: V = Q · Z
    let mut eigenvectors = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += q[i * n + k] * z[k * n + j];
            }
            eigenvectors[i * n + j] = s;
        }
    }

    // Sort ascending
    let mut indexed: Vec<(usize, f64)> = eigenvalues.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_vals: Vec<f64> = indexed.iter().map(|(_, v)| *v).collect();
    let mut sorted_vecs = vec![0.0; n * n];
    for (new_j, (old_j, _)) in indexed.iter().enumerate() {
        for i in 0..n {
            sorted_vecs[i * n + new_j] = eigenvectors[i * n + old_j];
        }
    }

    EighDecompositionF64 {
        eigenvalues: sorted_vals,
        eigenvectors: sorted_vecs,
        n,
    }
}

/// Analytic 2×2 eigensolver (no iteration needed).
fn eigh_2x2(a: &[f64]) -> EighDecompositionF64 {
    let a00 = a[0];
    let a01 = a[1];
    let a11 = a[3];

    let trace = a00 + a11;
    let det = a00 * a11 - a01 * a01;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

    let l0 = (trace - disc) * 0.5;
    let l1 = (trace + disc) * 0.5;

    let (v0, v1) = if a01.abs() > 1e-300 {
        let v0x = a01;
        let v0y = l0 - a00;
        let norm0 = v0x.hypot(v0y);
        let v1x = a01;
        let v1y = l1 - a00;
        let norm1 = v1x.hypot(v1y);
        ([v0x / norm0, v0y / norm0], [v1x / norm1, v1y / norm1])
    } else if a00 <= a11 {
        ([1.0, 0.0], [0.0, 1.0])
    } else {
        ([0.0, 1.0], [1.0, 0.0])
    };

    EighDecompositionF64 {
        eigenvalues: vec![l0, l1],
        eigenvectors: vec![v0[0], v1[0], v0[1], v1[1]],
        n: 2,
    }
}

/// Householder tridiagonalization: A = Q T Qᵀ.
///
/// Returns (diagonal, off-diagonal, Q) where T has `diag[i]` on diagonal
/// and `off_diag[i]` on the super/sub-diagonal (length n-1).
#[must_use]
pub fn householder_tridiag(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut work = a.to_vec();

    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }

    for k in 0..n.saturating_sub(2) {
        let m = n - k - 1;
        let mut x = vec![0.0; m];
        for i in 0..m {
            x[i] = work[(k + 1 + i) * n + k];
        }

        let sigma: f64 = x.iter().map(|xi| xi * xi).sum();
        let norm_x = sigma.sqrt();

        if norm_x < 1e-300 {
            continue;
        }

        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        let v0 = x[0] + sign * norm_x;
        let mut v = x;
        v[0] = v0;
        let norm_v_sq: f64 = v.iter().map(|vi| vi * vi).sum();
        let tau = 2.0 / norm_v_sq;

        // H * work (rows k+1..n)
        for j in k..n {
            let mut dot = 0.0;
            for i in 0..m {
                dot += v[i] * work[(k + 1 + i) * n + j];
            }
            for i in 0..m {
                work[(k + 1 + i) * n + j] -= tau * v[i] * dot;
            }
        }

        // work * H (columns k+1..n)
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..m {
                dot += work[i * n + k + 1 + j] * v[j];
            }
            for j in 0..m {
                work[i * n + k + 1 + j] -= tau * dot * v[j];
            }
        }

        // Q = Q * H (columns k+1..n)
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..m {
                dot += q[i * n + k + 1 + j] * v[j];
            }
            for j in 0..m {
                q[i * n + k + 1 + j] -= tau * dot * v[j];
            }
        }
    }

    let mut diag = vec![0.0; n];
    let mut off_diag = vec![0.0; n.saturating_sub(1)];
    for i in 0..n {
        diag[i] = work[i * n + i];
    }
    for i in 0..n.saturating_sub(1) {
        off_diag[i] = work[i * n + i + 1];
    }

    (diag, off_diag, q)
}

/// QL algorithm with implicit shifts for symmetric tridiagonal matrices.
///
/// Standard algorithm (cf. LAPACK DSTEQR / Numerical Recipes tql2).
/// Converges quadratically with Wilkinson shift.
fn ql_implicit(diag_in: &[f64], off_in: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d = diag_in.to_vec();
    let mut e = vec![0.0; n];
    e[..n - 1].copy_from_slice(&off_in[..n - 1]);

    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }

    for l in 0..n {
        let mut iter_count = 0_usize;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }

            if m == l {
                break;
            }

            iter_count += 1;
            if iter_count > 60 {
                break;
            }

            // Wilkinson shift
            let g0 = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r0 = g0.hypot(1.0);
            let g1 = d[m] - d[l] + e[l] / (g0 + r0.copysign(g0));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut g = g1;

            let mut converged_early = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r.abs() < 1e-300 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    converged_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                let g_new = d[i + 1] - p;
                let r2 = (d[i] - g_new) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_new + p;
                g = c * r2 - b;

                for k in 0..n {
                    let zk1 = z[k * n + i + 1];
                    let zk0 = z[k * n + i];
                    z[k * n + i + 1] = s * zk0 + c * zk1;
                    z[k * n + i] = c * zk0 - s * zk1;
                }
            }

            if converged_early {
                continue;
            }

            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    (d, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn uniform(&mut self) -> f64 {
            self.next_f64()
        }
    }

    fn max_off_diag(m: &[f64], n: usize) -> f64 {
        let mut mx = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    mx = mx.max(m[i * n + j].abs());
                }
            }
        }
        mx
    }

    fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = TestRng::new(seed);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rng.uniform() * 10.0 - 5.0;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn test_eigh_f64_2x2_simple() {
        let a = [3.0, 1.0, 1.0, 3.0];
        let r = eigh_householder_qr(&a, 2);
        assert!((r.eigenvalues[0] - 2.0).abs() < 1e-12);
        assert!((r.eigenvalues[1] - 4.0).abs() < 1e-12);
        assert!(r.reconstruction_error(&a) < 1e-12);
    }

    #[test]
    fn test_eigh_f64_3x3_tridiagonal() {
        let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let r = eigh_householder_qr(&a, 3);
        let expected = [
            2.0 - std::f64::consts::SQRT_2,
            2.0,
            2.0 + std::f64::consts::SQRT_2,
        ];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (r.eigenvalues[i] - exp).abs() < 1e-12,
                "eigenvalue[{i}]: got {}, expected {exp}",
                r.eigenvalues[i]
            );
        }
        assert!(r.reconstruction_error(&a) < 1e-12);
    }

    #[test]
    fn test_eigh_f64_householder_tridiag_structure() {
        let n = 5;
        let a = random_symmetric(n, 42);
        let (d, e, q) = householder_tridiag(&a, n);

        let mut t = vec![0.0; n * n];
        for i in 0..n {
            t[i * n + i] = d[i];
        }
        for i in 0..n - 1 {
            t[i * n + i + 1] = e[i];
            t[(i + 1) * n + i] = e[i];
        }

        // Q T Qᵀ ≈ A
        let mut qtq = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        s += q[i * n + k] * t[k * n + l] * q[j * n + l];
                    }
                }
                qtq[i * n + j] = s;
            }
        }

        let err: f64 = a
            .iter()
            .zip(qtq.iter())
            .map(|(ai, qi)| (ai - qi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err < 1e-10, "Q T Qᵀ reconstruction error: {err}");
    }

    #[test]
    fn test_eigh_f64_8x8_accuracy() {
        let a = random_symmetric(8, 42);
        let r = eigh_householder_qr(&a, 8);
        let err = r.reconstruction_error(&a);
        assert!(err < 1e-10, "n=8 reconstruction error: {err}");
    }

    #[test]
    fn test_eigh_f64_16x16_accuracy() {
        let a = random_symmetric(16, 123);
        let r = eigh_householder_qr(&a, 16);
        let err = r.reconstruction_error(&a);
        assert!(err < 1e-8, "n=16 reconstruction error: {err}");
    }

    #[test]
    fn test_eigh_f64_32x32_accuracy() {
        let a = random_symmetric(32, 999);
        let r = eigh_householder_qr(&a, 32);
        let err = r.reconstruction_error(&a);
        assert!(err < 1e-6, "n=32 reconstruction error: {err}");
    }

    #[test]
    fn test_eigh_f64_orthogonality() {
        let n = 8;
        let a = random_symmetric(n, 42);
        let r = eigh_householder_qr(&a, n);

        // VᵀV ≈ I
        let mut vtv = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    vtv[i * n + j] += r.eigenvectors[k * n + i] * r.eigenvectors[k * n + j];
                }
            }
        }

        let off_err = max_off_diag(&vtv, n);
        assert!(off_err < 1e-10, "VᵀV off-diagonal max: {off_err}");

        for i in 0..n {
            assert!(
                (vtv[i * n + i] - 1.0).abs() < 1e-10,
                "VᵀV diagonal[{i}]: {}",
                vtv[i * n + i]
            );
        }
    }

    #[test]
    fn test_eigh_f64_1x1() {
        let a = [42.0];
        let r = eigh_householder_qr(&a, 1);
        assert!((r.eigenvalues[0] - 42.0).abs() < 1e-14);
        assert!((r.eigenvectors[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_eigh_f64_identity() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let r = eigh_householder_qr(&a, n);
        for &ev in &r.eigenvalues {
            assert!((ev - 1.0).abs() < 1e-12);
        }
        assert!(r.reconstruction_error(&a) < 1e-12);
    }
}
