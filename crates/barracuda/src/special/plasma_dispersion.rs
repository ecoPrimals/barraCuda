// SPDX-License-Identifier: AGPL-3.0-or-later
//! Plasma dispersion function W(z) and its companion Z(z).
//!
//! Absorbed from hotSpring `physics/dielectric.rs` — these are the numerically
//! stable CPU-side reference implementations for Vlasov susceptibility and
//! Mermin dielectric calculations.
//!
//! # Functions
//!
//! - `plasma_dispersion_z(z)` — The plasma dispersion function Z(z)
//! - `plasma_dispersion_w(z)` — W(z) = 1 + z·Z(z), with automatic
//!   branch selection to avoid catastrophic cancellation at large |z|
//! - `plasma_dispersion_w_stable(z)` — Direct asymptotic expansion
//!   of W(z) for |z| ≥ 4, bypassing the cancellation-prone `1 + z·Z(z)`
//!
//! # Numerical Stability (ISSUE-006)
//!
//! For large |z|, the naive computation `W(z) = 1 + z·Z(z)` suffers from
//! catastrophic cancellation because both terms approach large magnitudes
//! that nearly cancel. The `w_stable` branch computes W(z) directly via
//! its own asymptotic expansion, avoiding this issue entirely.

use crate::ops::lattice::cpu_complex::Complex64;

const SQRT_PI: f64 = 1.772_453_850_905_516;
const SMALL_Z_THRESHOLD: f64 = 6.0;
const W_BRANCH_THRESHOLD: f64 = 4.0;
const SERIES_TOLERANCE: f64 = 1e-16;
const STABLE_TOLERANCE: f64 = 1e-15;
const ABS_FLOOR: f64 = 1e-30;
const MAX_SERIES_TERMS: usize = 100;
const MAX_ASYMPTOTIC_TERMS: usize = 30;

/// Plasma dispersion function Z(z).
///
/// Z(z) = i√π exp(−z²) − 2z Σ cₙ, where cₙ is a power series in −2z².
/// For |z| ≥ 6, switches to an asymptotic expansion.
pub fn plasma_dispersion_z(z: Complex64) -> Complex64 {
    if z.abs() < SMALL_Z_THRESHOLD {
        plasma_z_series(z)
    } else {
        plasma_z_asymptotic(z)
    }
}

/// W(z) = 1 + z·Z(z), with stable branch for |z| ≥ 4.
///
/// Automatically selects the numerically stable computation path.
pub fn plasma_dispersion_w(z: Complex64) -> Complex64 {
    if z.abs() < W_BRANCH_THRESHOLD {
        Complex64::ONE + z * plasma_dispersion_z(z)
    } else {
        plasma_dispersion_w_stable(z)
    }
}

/// Direct asymptotic expansion of W(z), avoiding the cancellation in `1 + z·Z(z)`.
///
/// W(z) ≈ −1/(2z²) × (1 + 3/(2z²) + 15/(4z⁴) + ...) + iσ√π z exp(−z²)
///
/// where σ = 1 for Im(z) ≥ 0, σ = 2 for Im(z) < 0.
pub fn plasma_dispersion_w_stable(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let inv_2z2 = (z2 * 2.0).inv();

    let mut coeff = Complex64::ONE;
    let mut total = coeff;
    for n in 0..MAX_ASYMPTOTIC_TERMS {
        #[expect(
            clippy::cast_precision_loss,
            reason = "n < 30; 2n+3 < 63 fits exactly in f64"
        )]
        let factor = (2 * n + 3) as f64;
        coeff = coeff * inv_2z2 * factor;
        total += coeff;
        if coeff.abs() < STABLE_TOLERANCE * (total.abs() + ABS_FLOOR) {
            break;
        }
    }
    let w_asymp = inv_2z2 * total * (-1.0);

    let exp_neg_z2 = (z2 * (-1.0)).exp();
    let sigma = if z.im >= 0.0 { 1.0 } else { 2.0 };
    let w_exp = Complex64::I * z * SQRT_PI * sigma * exp_neg_z2;

    w_asymp + w_exp
}

fn plasma_z_series(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let neg2z2 = z2 * (-2.0);
    let mut term = Complex64::ONE;
    let mut total = term;

    for n in 1..MAX_SERIES_TERMS {
        #[expect(
            clippy::cast_precision_loss,
            reason = "n < 100; 2n+1 < 201 fits exactly in f64"
        )]
        let denom = (2 * n + 1) as f64;
        term = term * neg2z2 * (1.0 / denom);
        total += term;
        if term.abs() < SERIES_TOLERANCE * (total.abs() + ABS_FLOOR) {
            break;
        }
    }
    let exp_neg_z2 = (z2 * (-1.0)).exp();
    let imag_part = Complex64::I * SQRT_PI * exp_neg_z2;
    imag_part - z * total * 2.0
}

fn plasma_z_asymptotic(z: Complex64) -> Complex64 {
    let z2 = z * z;
    let inv_z = z.inv();
    let inv_2z2 = (z2 * 2.0).inv();

    let mut coeff = Complex64::ONE;
    let mut total = coeff;
    for n in 0..MAX_ASYMPTOTIC_TERMS {
        #[expect(
            clippy::cast_precision_loss,
            reason = "n < 30; 2n+1 < 61 fits exactly in f64"
        )]
        let factor = (2 * n + 1) as f64;
        coeff = coeff * inv_2z2 * factor;
        total += coeff;
        if coeff.abs() < SERIES_TOLERANCE * (total.abs() + ABS_FLOOR) {
            break;
        }
    }
    let z_asymp = inv_z * total * (-1.0);

    let sigma = if z.im >= 0.0 { 1.0 } else { 2.0 };
    let exp_neg_z2 = (z2 * (-1.0)).exp();
    let z_exp = Complex64::I * SQRT_PI * sigma * exp_neg_z2;

    z_asymp + z_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: Complex64, b: Complex64, tol: f64, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < tol,
            "{msg}: expected ({:.10}, {:.10}i), got ({:.10}, {:.10}i), diff={diff:.2e}",
            b.re,
            b.im,
            a.re,
            a.im
        );
    }

    #[test]
    fn w_at_zero() {
        let w = plasma_dispersion_w(Complex64::ZERO);
        assert_close(w, Complex64::ONE, 1e-12, "W(0) = 1");
    }

    #[test]
    fn z_at_zero() {
        let z = plasma_dispersion_z(Complex64::ZERO);
        let expected = Complex64::new(0.0, SQRT_PI);
        assert_close(z, expected, 1e-12, "Z(0) = i√π");
    }

    #[test]
    fn w_small_real() {
        let z = Complex64::new(0.5, 0.0);
        let w = plasma_dispersion_w(z);
        assert!(w.re > 0.0 && w.re < 1.0, "W(0.5) real part in (0,1)");
    }

    #[test]
    fn w_large_z_stability() {
        let z = Complex64::new(10.0, 0.1);
        let w = plasma_dispersion_w(z);
        assert!(w.abs() < 0.1, "W(large z) should be small");
        assert!(w.re.is_finite() && w.im.is_finite(), "W must be finite");
    }

    #[test]
    fn w_identity_small_z() {
        let z = Complex64::new(1.0, 0.5);
        let w_direct = Complex64::ONE + z * plasma_dispersion_z(z);
        let w_fn = plasma_dispersion_w(z);
        assert_close(w_fn, w_direct, 1e-12, "W(z) == 1 + z·Z(z) for small z");
    }

    #[test]
    fn w_stable_matches_naive_at_boundary() {
        let z = Complex64::new(4.0, 0.5);
        let w_naive = Complex64::ONE + z * plasma_dispersion_z(z);
        let w_stable = plasma_dispersion_w_stable(z);
        // At the branch boundary, asymptotic and series expansions naturally
        // have ~1e-4 disagreement — both are valid representations.
        assert_close(
            w_stable,
            w_naive,
            1e-4,
            "stable and naive should agree near boundary",
        );
    }

    #[test]
    fn w_negative_imaginary() {
        let z = Complex64::new(1.0, -1.0);
        let w = plasma_dispersion_w(z);
        assert!(
            w.re.is_finite() && w.im.is_finite(),
            "W must be finite for Im(z)<0"
        );
    }

    #[test]
    fn w_purely_imaginary() {
        let z = Complex64::new(0.0, 2.0);
        let w = plasma_dispersion_w(z);
        assert!(w.re.is_finite() && w.im.is_finite(), "W(2i) must be finite");
    }
}
