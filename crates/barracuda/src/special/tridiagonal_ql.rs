// SPDX-License-Identifier: AGPL-3.0-only
//! Symmetric tridiagonal QL eigensolver with implicit shifts.
//!
//! Computes all eigenvalues and eigenvectors of a real symmetric tridiagonal
//! matrix using Givens rotations with Wilkinson shifts. This is the same
//! algorithm used in LAPACK's `dsteqr`.
//!
//! # Use Cases
//!
//! - Anderson localization: `anderson_diagonalize(disorder, t_hop)`
//! - Lanczos completion: Lanczos reduces large sparse → tridiag; QL finishes
//! - Any symmetric tridiagonal eigenvalue problem
//!
//! Absorbed from healthSpring `microbiome.rs` (V13) and generalized for
//! all springs.

/// Maximum QL iterations per eigenvalue before giving up.
const MAX_QL_ITERATIONS: u32 = 200;

/// Near-zero threshold for detecting converged off-diagonal elements.
const UNDERFLOW_GUARD: f64 = 1e-300;

/// Diagonalize a symmetric tridiagonal matrix via the QL algorithm.
///
/// Given diagonal `d[0..n]` and sub-diagonal `e[0..n-1]`, computes:
/// - **eigenvalues** in `d` (modified in-place, returned)
/// - **eigenvectors** as columns of `z` (row-major `n × n`, returned)
///
/// The sub-diagonal `e` is destroyed on output.
///
/// # Arguments
///
/// * `diagonal` — Main diagonal elements (length n).
/// * `sub_diagonal` — Sub-diagonal elements (length n−1). Padded internally.
/// * Returns `(eigenvalues, eigenvectors)` where eigenvectors is row-major `n × n`.
#[must_use]
pub fn tridiagonal_ql(diagonal: &[f64], sub_diagonal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = diagonal.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (diagonal.to_vec(), vec![1.0]);
    }

    let mut d = diagonal.to_vec();
    // EISPACK tql2 convention: the shift `e[i-1] = e[i]` moves sub-diagonal
    // from positions 1..n-1 to 0..n-2, so we store at offset +1.
    let mut e = vec![0.0_f64; n];
    for (i, &val) in sub_diagonal.iter().enumerate().take(n - 1) {
        e[i + 1] = val;
    }

    let mut z = vec![0.0_f64; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }

    ql_implicit_shifts(&mut d, &mut e, &mut z, n);

    (d, z)
}

/// Diagonalize an Anderson tight-binding Hamiltonian.
///
/// The Anderson model has on-site disorder potentials on the diagonal and
/// uniform hopping amplitude `t_hop` on the sub-diagonal.
///
/// Returns `(eigenvalues, eigenvectors)`.
#[must_use]
pub fn anderson_diagonalize(disorder: &[f64], t_hop: f64) -> (Vec<f64>, Vec<f64>) {
    let n = disorder.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let sub_diag = vec![t_hop; n.saturating_sub(1)];
    tridiagonal_ql(disorder, &sub_diag)
}

/// QL algorithm with implicit Wilkinson shifts and Givens rotations.
fn ql_implicit_shifts(d: &mut [f64], e: &mut [f64], z: &mut [f64], n: usize) {
    // Shift sub-diagonal down by one position (standard QL convention).
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    for l in 0..n {
        let mut iter_count = 0_u32;
        loop {
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                #[expect(
                    clippy::float_cmp,
                    reason = "standard QL convergence: e[m] is negligible when adding it to dd doesn't change dd at f64 precision"
                )]
                if (e[m].abs() + dd) == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter_count += 1;
            if iter_count > MAX_QL_ITERATIONS {
                break;
            }

            // Wilkinson shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r.abs() < UNDERFLOW_GUARD {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g).mul_add(s, 2.0 * c * b);
                p = s * r;
                d[i + 1] = g + p;
                g = c.mul_add(r, -b);

                // Accumulate eigenvector rotation
                for k in 0..n {
                    f = z[(i + 1) * n + k];
                    z[(i + 1) * n + k] = s.mul_add(z[i * n + k], c * f);
                    z[i * n + k] = c.mul_add(z[i * n + k], -(s * f));
                }
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let (vals, vecs) = tridiagonal_ql(&[], &[]);
        assert!(vals.is_empty());
        assert!(vecs.is_empty());
    }

    #[test]
    fn single_element() {
        let (vals, vecs) = tridiagonal_ql(&[3.0], &[]);
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 3.0).abs() < 1e-14);
        assert!((vecs[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn two_by_two() {
        // Matrix: [[2, 1], [1, 2]] → eigenvalues 1 and 3
        let (vals, vecs) = tridiagonal_ql(&[2.0, 2.0], &[1.0]);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12, "got {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-12, "got {}", sorted[1]);

        // Eigenvectors should be orthonormal
        let n = 2;
        for i in 0..n {
            let norm_sq: f64 = (0..n).map(|k| vecs[i * n + k].powi(2)).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-12,
                "eigenvector {i} not normalized: {norm_sq}"
            );
        }
    }

    #[test]
    fn anderson_3_site() {
        let disorder = [0.5, -0.3, 0.1];
        let t_hop = 1.0;
        let (vals, vecs) = anderson_diagonalize(&disorder, t_hop);
        assert_eq!(vals.len(), 3);
        assert_eq!(vecs.len(), 9);

        // Eigenvalues should sum to trace = 0.5 - 0.3 + 0.1 = 0.3
        let trace: f64 = vals.iter().sum();
        assert!(
            (trace - 0.3).abs() < 1e-10,
            "trace mismatch: {trace} vs 0.3"
        );

        // Each eigenvector should be normalized
        for i in 0..3 {
            let norm_sq: f64 = (0..3).map(|k| vecs[i * 3 + k].powi(2)).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-10,
                "eigenvector {i} not normalized: {norm_sq}"
            );
        }
    }

    #[test]
    fn diagonal_matrix() {
        let diag = [5.0, 2.0, 8.0, 1.0];
        let sub_diag = [0.0, 0.0, 0.0];
        let (vals, _) = tridiagonal_ql(&diag, &sub_diag);

        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut expected = diag.to_vec();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (v, e) in sorted.iter().zip(expected.iter()) {
            assert!((v - e).abs() < 1e-12, "eigenvalue {v} != {e}");
        }
    }

    #[test]
    fn eigenvectors_orthogonal() {
        let disorder: Vec<f64> = (0..10).map(|i| (i as f64 * 0.7).sin()).collect();
        let (_, vecs) = anderson_diagonalize(&disorder, 1.0);
        let n = 10;

        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f64 = (0..n).map(|k| vecs[i * n + k] * vecs[j * n + k]).sum();
                assert!(
                    dot.abs() < 1e-10,
                    "eigenvectors {i} and {j} not orthogonal: dot={dot}"
                );
            }
        }
    }
}
