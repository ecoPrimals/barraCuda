// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hofstadter butterfly and Almost-Mathieu (Harper) operator.
//!
//! Quasiperiodic potential with Aubry-André transition; spectral topology.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)

use super::tridiag::find_all_eigenvalues;

/// Construct the almost-Mathieu (Harper) operator on N sites:
///   H `ψ_n` = ψ_{n+1} + ψ_{n-1} + 2λ cos(2παn + θ) `ψ_n`
///
/// Returns (diagonal, `off_diagonal`) for the tridiagonal representation.
///
/// At the Aubry-André self-dual point λ = 1, the operator undergoes a
/// metal-insulator transition:
/// - λ < 1: absolutely continuous spectrum (extended states)
/// - λ > 1: pure point spectrum (localized states, Anderson-like)
/// - λ = 1: singular continuous spectrum (critical)
///
/// # Provenance
/// - Aubry & André (1980)
/// - Harper (1955), Proc. Phys. Soc. London A 68, 874
/// - Hofstadter (1976), Phys. Rev. B 14, 2239 (butterfly)
#[must_use]
pub fn almost_mathieu_hamiltonian(
    n: usize,
    lambda: f64,
    alpha: f64,
    theta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let diagonal: Vec<f64> = (0..n)
        .map(|i| 2.0 * lambda * (std::f64::consts::TAU * alpha * i as f64 + theta).cos())
        .collect();
    let off_diag = vec![1.0; n - 1];

    (diagonal, off_diag)
}

/// The golden ratio (√5 − 1)/2 — the canonical choice of irrational
/// frequency for the almost-Mathieu operator.
pub const GOLDEN_RATIO: f64 = 0.618_033_988_749_894_9;

/// Compute the Hofstadter butterfly: spectrum of the almost-Mathieu operator
/// as a function of magnetic flux α.
///
/// For each rational flux α = p/q with q ≤ `q_max` and gcd(p,q) = 1,
/// computes all eigenvalues of a large almost-Mathieu operator. The
/// resulting (α, E) point cloud forms the Hofstadter butterfly — a fractal
/// spectrum that is the canonical example of spectral topology.
///
/// At λ = 1 (critical coupling), the spectrum at irrational α is a Cantor
/// set of measure zero (the "Ten Martini Problem", proved by Avila &
/// Jitomirskaya 2009).
///
/// Returns: Vec of (alpha, eigenvalues) pairs.
///
/// # Provenance
/// Hofstadter (1976) Phys. Rev. B 14, 2239
/// Avila & Jitomirskaya (2009) Ann. Math. 170, 303
#[must_use]
pub fn hofstadter_butterfly(q_max: usize, lambda: f64, n_sites: usize) -> Vec<(f64, Vec<f64>)> {
    let mut results = Vec::new();

    for q in 1..=q_max {
        for p in 1..q {
            if gcd(p, q) != 1 {
                continue;
            }
            let alpha = p as f64 / q as f64;
            let (d, e) = almost_mathieu_hamiltonian(n_sites, lambda, alpha, 0.0);
            let evals = find_all_eigenvalues(&d, &e);
            results.push((alpha, evals));
        }
    }

    results
}

/// Greatest common divisor (Euclid's algorithm).
#[must_use]
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn almost_mathieu_spectrum_bounds() {
        let lambda = 1.5;
        let n = 500;
        let (d, e) = almost_mathieu_hamiltonian(n, lambda, GOLDEN_RATIO, 0.0);
        let evals = find_all_eigenvalues(&d, &e);
        let bound = 2.0 + 2.0 * lambda + 0.01;
        for &ev in &evals {
            assert!(
                ev.abs() <= bound,
                "eigenvalue {ev:.4} outside [-{bound:.2}, {bound:.2}]"
            );
        }
    }

    #[test]
    #[expect(clippy::float_cmp, reason = "tests")]
    fn almost_mathieu_lambda_zero_free_lattice() {
        let (d, e) = almost_mathieu_hamiltonian(50, 0.0, GOLDEN_RATIO, 0.0);
        assert!(d.iter().all(|&x| x == 0.0));
        assert_eq!(e.len(), 49);
        assert!(e.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn hofstadter_butterfly_q_max_two() {
        let results = hofstadter_butterfly(2, 1.0, 50);
        assert!(!results.is_empty());
        assert_eq!(results.len(), 1);
        assert!((results[0].0 - 0.5).abs() < 1e-10);
        assert_eq!(results[0].1.len(), 50);
    }

    #[test]
    fn gcd_coprime_large() {
        assert_eq!(gcd(97, 89), 1);
    }

    #[test]
    fn gcd_same_number() {
        assert_eq!(gcd(42, 42), 42);
    }
}
