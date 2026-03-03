// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error function and complementary error function
//!
//! Implements erf(x) and erfc(x) using rational polynomial approximations
//! from Abramowitz & Stegun.
//!
//! # Precision
//!
//! |ε| < 1.5e-7 for all x (A&S 7.1.26)
//!
//! # References
//!
//! - Abramowitz & Stegun, §7.1
//! - DLMF 7.2: <https://dlmf.nist.gov/7.2>

/// WGSL kernel for complementary error function derivative (f64).
pub const WGSL_ERFC_DERIV_F64: &str = include_str!("../shaders/special/erfc_deriv_f64.wgsl");

/// Compute the error function erf(x).
///
/// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// # Properties
///
/// - erf(0) = 0
/// - erf(∞) = 1
/// - erf(-x) = -erf(x) (odd function)
///
/// # Precision
///
/// |ε| < 1.5e-7 (Abramowitz & Stegun 7.1.26)
///
/// # Examples
///
/// ```
/// use barracuda::special::erf;
///
/// assert!((erf(0.0) - 0.0).abs() < 1e-14);
/// assert!((erf(1.0) - 0.8427007929).abs() < 2e-7);
/// assert!((erf(-1.0) + 0.8427007929).abs() < 2e-7);  // Odd function
/// ```
pub fn erf(x: f64) -> f64 {
    // Special cases
    if x == 0.0 {
        return 0.0;
    }

    // Use symmetry for negative x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz & Stegun 7.1.26 approximation
    // |ε| < 1.5e-7
    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let result = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();

    sign * result
}

/// Compute the complementary error function erfc(x) = 1 - erf(x).
///
/// More accurate than computing `1 - erf(x)` for large x due to
/// catastrophic cancellation.
///
/// # Properties
///
/// - erfc(0) = 1
/// - erfc(∞) = 0
/// - erfc(-∞) = 2
///
/// # Precision
///
/// |ε| < 1.5e-7 for x ≥ 0
///
/// # Examples
///
/// ```
/// use barracuda::special::erfc;
///
/// assert!((erfc(0.0) - 1.0).abs() < 1e-14);
/// assert!((erfc(3.0) - 2.209049699858544e-5).abs() < 1e-5);
/// ```
pub fn erfc(x: f64) -> f64 {
    // Special case
    if x == 0.0 {
        return 1.0;
    }

    // For large positive x, compute directly for accuracy
    if x > 0.0 {
        let p = 0.3275911;
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;

        let t = 1.0 / (1.0 + p * x);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;

        (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp()
    } else {
        1.0 - erf(x)
    }
}

/// Compute erf for a batch of values (CPU path).
///
/// For GPU batch processing, use `ErfGpu` from `barracuda::special`.
pub fn erf_batch(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| erf(v)).collect()
}

/// Compute erfc for a batch of values (CPU path).
pub fn erfc_batch(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| erfc(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_zero() {
        assert!((erf(0.0) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_erf_one() {
        // scipy.special.erf(1.0) = 0.8427007929497149
        // A&S 7.1.26 precision: |ε| < 1.5e-7
        assert!((erf(1.0) - 0.8427007929497149).abs() < 2e-6);
    }

    #[test]
    fn test_erf_two() {
        // scipy.special.erf(2.0) = 0.9953222650189527
        assert!((erf(2.0) - 0.9953222650189527).abs() < 2e-6);
    }

    #[test]
    fn test_erf_negative() {
        // erf is an odd function
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-14);
        assert!((erf(-0.5) + erf(0.5)).abs() < 1e-14);
    }

    #[test]
    fn test_erf_large() {
        // erf(x) → 1 as x → ∞
        assert!((erf(5.0) - 1.0).abs() < 1e-7);
        assert!((erf(10.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_erfc_zero() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_erfc_large() {
        // scipy.special.erfc(3.0) = 2.209049699858544e-05
        // Relative error check (better for small values)
        let computed = erfc(3.0);
        let expected = 2.209049699858544e-5;
        assert!(
            (computed - expected).abs() / expected < 1e-3,
            "erfc(3) = {}, expected ~{}",
            computed,
            expected
        );
    }

    #[test]
    fn test_erf_erfc_relation() {
        for x in [0.0, 0.5, 1.0, 1.5, 2.0] {
            assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_erf_batch() {
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let result = erf_batch(&x);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-14);
        assert!((result[2] - 0.8427007929497149).abs() < 2e-6);
    }

    #[test]
    fn test_erf_special_values() {
        // erf(0.5) from scipy (A&S 7.1.26 precision: ~1.5e-7)
        assert!((erf(0.5) - 0.5204998778130465).abs() < 2e-6);
        // erf(1.5) from scipy
        assert!((erf(1.5) - 0.9661051464753108).abs() < 2e-6);
    }
}
