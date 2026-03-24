// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated numerically stable special functions.
//!
//! Cross-spring P1 (ISSUE-011): `log1p`, `expm1`, `erfc`, `bessel_j0_minus1`
//! implemented to avoid catastrophic cancellation.

/// CPU reference: `log(1 + x)` without cancellation for small x.
#[must_use]
pub fn log1p_f64(x: f64) -> f64 {
    x.ln_1p()
}

/// CPU reference: `exp(x) - 1` without cancellation for small x.
#[must_use]
pub fn expm1_f64(x: f64) -> f64 {
    x.exp_m1()
}

/// CPU reference: complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Uses Abramowitz & Stegun 7.1.26 rational approximation.
#[must_use]
pub fn erfc_f64(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_f64(-x);
    }
    let t = 1.0 / 0.327_591_1f64.mul_add(x, 1.0);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

/// CPU reference: `J₀(x) - 1` for small x via power series.
///
/// Avoids cancellation when `J₀(x) ≈ 1`.
#[must_use]
pub fn bessel_j0_minus1_f64(x: f64) -> f64 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x6 * x2;
    -x2 / 4.0 + x4 / 64.0 - x6 / 2304.0 + x8 / 147_456.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log1p_small() {
        let x = 1e-15;
        let result = log1p_f64(x);
        assert!((result - x).abs() < 1e-28, "log1p(1e-15) ≈ 1e-15");
    }

    #[test]
    fn test_log1p_moderate() {
        let x = 0.5;
        let result = log1p_f64(x);
        let expected = (1.5_f64).ln();
        assert!((result - expected).abs() < 1e-14);
    }

    #[test]
    fn test_expm1_small() {
        let x = 1e-15;
        let result = expm1_f64(x);
        assert!((result - x).abs() < 1e-28, "expm1(1e-15) ≈ 1e-15");
    }

    #[test]
    fn test_expm1_moderate() {
        let x = 0.5;
        let result = expm1_f64(x);
        let expected = (0.5_f64).exp_m1();
        assert!((result - expected).abs() < 1e-14);
    }

    #[test]
    fn test_erfc_zero() {
        let r = erfc_f64(0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erfc_large() {
        let r = erfc_f64(5.0);
        assert!(r > 0.0 && r < 1e-10, "erfc(5) should be tiny but positive");
    }

    #[test]
    fn test_erfc_negative() {
        let r = erfc_f64(-1.0);
        assert!(r > 1.0 && r < 2.0);
    }

    #[test]
    fn test_bessel_j0m1_zero() {
        let r = bessel_j0_minus1_f64(0.0);
        assert!(r.abs() < 1e-30, "J0(0)-1 = 0");
    }

    #[test]
    fn test_bessel_j0m1_small() {
        let r = bessel_j0_minus1_f64(0.1);
        let expected = -0.0025;
        assert!(
            (r - expected).abs() < 1e-5,
            "J0(0.1)-1 ≈ {r}, expected ≈ {expected}"
        );
    }
}
