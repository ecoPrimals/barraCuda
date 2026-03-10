// SPDX-License-Identifier: AGPL-3.0-only

//! Sturm bisection eigensolve for symmetric tridiagonal matrices.
//!
//! Counts eigenvalues below a given value using LDLT factorization (Sturm
//! sequence) and finds all eigenvalues via bisection.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)

/// Count eigenvalues of a symmetric tridiagonal matrix strictly less than λ.
///
/// Uses the LDLT factorization (Sturm sequence): the number of negative
/// pivots equals the number of eigenvalues below λ.
///
/// - `diagonal`: main diagonal d[0..n]
/// - `off_diag`: sub/super-diagonal e[0..n-1]
#[must_use]
pub fn sturm_count(diagonal: &[f64], off_diag: &[f64], lambda: f64) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }

    let mut count = 0;
    let mut q = diagonal[0] - lambda;
    if q < 0.0 {
        count += 1;
    }

    for i in 1..n {
        let q_safe = if q.abs() < 1e-300 {
            if q >= 0.0 { 1e-300 } else { -1e-300 }
        } else {
            q
        };
        q = (diagonal[i] - lambda) - off_diag[i - 1] * off_diag[i - 1] / q_safe;
        if q < 0.0 {
            count += 1;
        }
    }
    count
}

/// Find all eigenvalues of a symmetric tridiagonal matrix via Sturm bisection.
///
/// Returns eigenvalues sorted in ascending order. Complexity: O(N² log(1/ε)).
/// Exact to machine precision for well-separated eigenvalues.
#[must_use]
pub fn find_all_eigenvalues(diagonal: &[f64], off_diag: &[f64]) -> Vec<f64> {
    let n = diagonal.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![diagonal[0]];
    }

    // Gershgorin bounds
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for i in 0..n {
        let e_left = if i > 0 { off_diag[i - 1].abs() } else { 0.0 };
        let e_right = if i < n - 1 { off_diag[i].abs() } else { 0.0 };
        lo = lo.min(diagonal[i] - e_left - e_right);
        hi = hi.max(diagonal[i] + e_left + e_right);
    }
    lo -= 1.0;
    hi += 1.0;

    let mut eigenvalues = Vec::with_capacity(n);
    for k in 0..n {
        let mut a = lo;
        let mut b = hi;
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if (b - a) < 2.0 * f64::EPSILON * mid.abs().max(1.0) {
                break;
            }
            if sturm_count(diagonal, off_diag, mid) <= k {
                a = mid;
            } else {
                b = mid;
            }
        }
        eigenvalues.push(0.5 * (a + b));
    }
    eigenvalues
}

/// Find all eigenvalues AND eigenvectors of a symmetric tridiagonal matrix.
///
/// Uses Sturm bisection for eigenvalues, then inverse iteration for each
/// eigenvector. Returns `(eigenvalues, eigenvectors)` where eigenvectors
/// is an n×n column-major matrix: `eigenvectors[i * n + k]` is the i-th
/// component of the k-th eigenvector.
///
/// Complexity: O(N² log(1/ε)) for eigenvalues + O(N²) per eigenvector.
/// For degenerate eigenvalues, the vectors are orthogonalized via
/// modified Gram-Schmidt.
#[must_use]
pub fn tridiag_eigenvectors(diagonal: &[f64], off_diag: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = diagonal.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let evals = find_all_eigenvalues(diagonal, off_diag);
    if n == 1 {
        return (evals, vec![1.0]);
    }

    let mut vecs = vec![0.0; n * n];

    for k in 0..n {
        let lambda = evals[k];
        let v = inverse_iteration_tridiag(diagonal, off_diag, lambda);

        // Orthogonalize against previous eigenvectors (Gram-Schmidt)
        let mut orth = v;
        for prev in 0..k {
            let dot: f64 = (0..n).map(|i| orth[i] * vecs[i * n + prev]).sum();
            for i in 0..n {
                orth[i] -= dot * vecs[i * n + prev];
            }
        }

        let norm: f64 = orth.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in &mut orth {
                *x /= norm;
            }
        }

        for i in 0..n {
            vecs[i * n + k] = orth[i];
        }
    }

    (evals, vecs)
}

/// Inverse iteration for a single eigenvector of a tridiagonal matrix.
///
/// Solves (T - λI)x = b repeatedly via an LU factorization of the shifted
/// tridiagonal. Converges in O(1) iterations for well-separated eigenvalues.
fn inverse_iteration_tridiag(diagonal: &[f64], off_diag: &[f64], lambda: f64) -> Vec<f64> {
    let n = diagonal.len();

    // LU factorize (T - λI) once, then repeatedly solve LU x = b.
    // For a symmetric tridiagonal with diagonal a_i = diag[i] - λ
    // and off-diagonal e_i = off_diag[i], the LU decomposition is:
    //   L = bidiag(l[i], 1)  U = bidiag(u[i], e[i])
    //   where u[0] = a[0], l[i] = e[i-1]/u[i-1], u[i] = a[i] - l[i]*e[i-1]

    let mut u = vec![0.0; n];
    let mut l = vec![0.0; n]; // l[0] unused
    u[0] = diagonal[0] - lambda;
    if u[0].abs() < 1e-300 {
        u[0] = 1e-300;
    }

    for i in 1..n {
        l[i] = off_diag[i - 1] / u[i - 1];
        u[i] = (diagonal[i] - lambda) - l[i] * off_diag[i - 1];
        if u[i].abs() < 1e-300 {
            u[i] = 1e-300;
        }
    }

    let mut x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64)).collect();

    for _ in 0..10 {
        // Forward solve: L y = x  (y stored in-place)
        let mut y = x.clone();
        for i in 1..n {
            y[i] -= l[i] * y[i - 1];
        }
        // Back solve: U x_new = y
        x[n - 1] = y[n - 1] / u[n - 1];
        for i in (0..n - 1).rev() {
            x[i] = (y[i] - off_diag[i] * x[i + 1]) / u[i];
        }
        // Normalize
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-300 {
            break;
        }
        for v in &mut x {
            *v /= norm;
        }
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sturm_count_identity_2x2() {
        let d = [1.0, 3.0];
        let e = [-1.0];
        assert_eq!(sturm_count(&d, &e, 0.0), 0);
        assert_eq!(sturm_count(&d, &e, 1.0), 1);
        assert_eq!(sturm_count(&d, &e, 4.0), 2);
    }

    #[test]
    fn eigenvalues_clean_chain() {
        let n = 50;
        let d = vec![0.0; n];
        let e = vec![-1.0; n - 1];
        let evals = find_all_eigenvalues(&d, &e);

        assert_eq!(evals.len(), n);

        for k in 1..=n {
            let exact = 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            let closest = evals
                .iter()
                .map(|&ev| (ev - exact).abs())
                .fold(f64::MAX, f64::min);
            assert!(
                closest < 1e-10,
                "k={k}, exact={exact:.6}, closest error={closest:.2e}"
            );
        }
    }

    #[test]
    fn eigenvectors_orthonormal_3x3() {
        // diag(2, 3, 4) with off-diag -1: known tridiag
        let d = vec![2.0, 3.0, 4.0];
        let e = vec![-1.0, -1.0];
        let (evals, vecs) = tridiag_eigenvectors(&d, &e);

        assert_eq!(evals.len(), 3);
        assert_eq!(vecs.len(), 9);

        // Verify orthonormality: V^T V = I
        for k in 0..3 {
            for l in 0..3 {
                let dot: f64 = (0..3).map(|i| vecs[i * 3 + k] * vecs[i * 3 + l]).sum();
                if k == l {
                    assert!(
                        (dot - 1.0).abs() < 1e-10,
                        "v[{k}]·v[{l}] = {dot}, expected 1"
                    );
                } else {
                    assert!((dot).abs() < 1e-10, "v[{k}]·v[{l}] = {dot}, expected 0");
                }
            }
        }

        // Verify T * v_k = λ_k * v_k for each pair
        for k in 0..3 {
            let v: Vec<f64> = (0..3).map(|i| vecs[i * 3 + k]).collect();
            // T*v
            let mut tv = [0.0; 3];
            tv[0] = d[0] * v[0] + e[0] * v[1];
            tv[1] = e[0] * v[0] + d[1] * v[1] + e[1] * v[2];
            tv[2] = e[1] * v[1] + d[2] * v[2];

            for i in 0..3 {
                let diff = (tv[i] - evals[k] * v[i]).abs();
                assert!(diff < 1e-8, "T*v[{k}][{i}] - λ*v[{k}][{i}] = {diff:.2e}");
            }
        }
    }

    #[test]
    fn eigenvectors_clean_chain_20() {
        let n = 20;
        let d = vec![0.0; n];
        let e = vec![-1.0; n - 1];
        let (evals, vecs) = tridiag_eigenvectors(&d, &e);

        assert_eq!(evals.len(), n);
        assert_eq!(vecs.len(), n * n);

        // Verify T*v = λ*v for a few eigenpairs
        for k in [0, n / 4, n / 2, 3 * n / 4, n - 1] {
            let v: Vec<f64> = (0..n).map(|i| vecs[i * n + k]).collect();
            let mut tv = vec![0.0; n];
            tv[0] = d[0] * v[0] + e[0] * v[1];
            for i in 1..n - 1 {
                tv[i] = e[i - 1] * v[i - 1] + d[i] * v[i] + e[i] * v[i + 1];
            }
            tv[n - 1] = e[n - 2] * v[n - 2] + d[n - 1] * v[n - 1];

            let max_err: f64 = (0..n)
                .map(|i| (tv[i] - evals[k] * v[i]).abs())
                .fold(0.0, f64::max);
            assert!(
                max_err < 1e-8,
                "eigvec k={k}: max |T*v - λ*v| = {max_err:.2e}"
            );
        }
    }

    #[test]
    fn eigenvectors_singleton() {
        let (evals, vecs) = tridiag_eigenvectors(&[5.0], &[]);
        assert_eq!(evals, vec![5.0]);
        assert_eq!(vecs, vec![1.0]);
    }

    #[test]
    fn eigenvectors_empty() {
        let (evals, vecs) = tridiag_eigenvectors(&[], &[]);
        assert!(evals.is_empty());
        assert!(vecs.is_empty());
    }
}
