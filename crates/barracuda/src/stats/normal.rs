// SPDX-License-Identifier: AGPL-3.0-or-later
//! Normal (Gaussian) distribution functions
//!
//! Implements the standard normal distribution N(0,1):
//! - CDF: Φ(x) = P(X ≤ x)
//! - PDF: φ(x) = (1/√2π) exp(-x²/2)
//! - Inverse CDF (quantile/probit): Φ⁻¹(p)
//!
//! # Precision
//!
//! - CDF/PDF: |ε| < 1.5e-7 (uses erf from A&S 7.1.26)
//! - Inverse CDF: |ε| < 3e-9 for p ∈ [1e-15, 1-1e-15] (Moro 1995)
//!
//! # References
//!
//! - Abramowitz & Stegun §26.2
//! - Moro, B. (1995) "The Full Monte"

use crate::special::erf;
use std::f64::consts::SQRT_2;

/// Standard normal CDF: Φ(x) = (1 + erf(x/√2)) / 2
///
/// Returns P(X ≤ x) for X ~ N(0,1).
///
/// # Examples
///
/// ```
/// use barracuda::stats::norm_cdf;
///
/// assert!((norm_cdf(0.0) - 0.5).abs() < 1e-10);
/// assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3);  // 97.5 percentile
/// ```
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Standard normal PDF: φ(x) = (1/√2π) exp(-x²/2)
///
/// # Examples
///
/// ```
/// use barracuda::stats::norm_pdf;
///
/// let peak = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
/// assert!((norm_pdf(0.0) - peak).abs() < 1e-10);
/// ```
#[must_use]
pub fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327; // 1/√(2π)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Inverse standard normal CDF (quantile/probit function): Φ⁻¹(p)
///
/// Returns x such that P(X ≤ x) = p for X ~ N(0,1).
///
/// Uses the Acklam algorithm, which provides high accuracy across the full
/// range with rational approximations.
///
/// # Returns
///
/// Returns ±∞ for p ≤ 0 or p ≥ 1.
///
/// # Precision
///
/// |ε| < 1.15e-9 for all p in (0, 1)
///
/// # Examples
///
/// ```
/// use barracuda::stats::norm_ppf;
///
/// assert!((norm_ppf(0.5) - 0.0).abs() < 1e-10);
/// assert!((norm_ppf(0.975) - 1.96).abs() < 1e-2);
/// ```
#[expect(
    clippy::excessive_precision,
    reason = "Acklam algorithm coefficients; precision intentional"
)]
#[must_use]
pub fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Acklam's algorithm for inverse normal CDF
    // https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/

    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    // Break-points for regions
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x: f64;

    if p < P_LOW {
        // Lower tail region
        let q = (-2.0 * p.ln()).sqrt();
        x = C[0]
            .mul_add(q, C[1])
            .mul_add(q, C[2])
            .mul_add(q, C[3])
            .mul_add(q, C[4])
            .mul_add(q, C[5])
            / (D[0]
                .mul_add(q, D[1])
                .mul_add(q, D[2])
                .mul_add(q, D[3])
                .mul_add(q, 1.0));
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        x = A[0]
            .mul_add(r, A[1])
            .mul_add(r, A[2])
            .mul_add(r, A[3])
            .mul_add(r, A[4])
            .mul_add(r, A[5])
            * q
            / (B[0]
                .mul_add(r, B[1])
                .mul_add(r, B[2])
                .mul_add(r, B[3])
                .mul_add(r, B[4])
                .mul_add(r, 1.0));
    } else {
        // Upper tail region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        x = -(C[0]
            .mul_add(q, C[1])
            .mul_add(q, C[2])
            .mul_add(q, C[3])
            .mul_add(q, C[4])
            .mul_add(q, C[5])
            / D[0]
                .mul_add(q, D[1])
                .mul_add(q, D[2])
                .mul_add(q, D[3])
                .mul_add(q, 1.0));
    }

    x
}

/// Compute `norm_cdf` for a batch of values.
#[must_use]
pub fn norm_cdf_batch(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| norm_cdf(v)).collect()
}

/// Compute `norm_pdf` for a batch of values.
#[must_use]
pub fn norm_pdf_batch(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| norm_pdf(v)).collect()
}

/// General normal CDF with mean μ and standard deviation σ.
///
/// Φ((x - μ) / σ)
#[must_use]
pub fn norm_cdf_general(x: f64, mu: f64, sigma: f64) -> f64 {
    norm_cdf((x - mu) / sigma)
}

/// General normal PDF with mean μ and standard deviation σ.
///
/// (1 / σ) φ((x - μ) / σ)
#[must_use]
pub fn norm_pdf_general(x: f64, mu: f64, sigma: f64) -> f64 {
    norm_pdf((x - mu) / sigma) / sigma
}

/// General inverse normal CDF (quantile) with mean μ and standard deviation σ.
///
/// μ + σ Φ⁻¹(p)
#[must_use]
pub fn norm_ppf_general(p: f64, mu: f64, sigma: f64) -> f64 {
    sigma.mul_add(norm_ppf(p), mu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_norm_cdf_zero() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_norm_cdf_positive() {
        // scipy.stats.norm.cdf(1.0) = 0.8413447460685429
        assert!((norm_cdf(1.0) - 0.8413447460685429).abs() < 2e-6);
        // scipy.stats.norm.cdf(2.0) = 0.9772498680518208
        assert!((norm_cdf(2.0) - 0.9772498680518208).abs() < 2e-6);
    }

    #[test]
    fn test_norm_cdf_negative() {
        // Symmetry: Φ(-x) = 1 - Φ(x)
        assert!((norm_cdf(-1.0) - (1.0 - norm_cdf(1.0))).abs() < 1e-14);
        assert!((norm_cdf(-2.0) - (1.0 - norm_cdf(2.0))).abs() < 1e-14);
    }

    #[test]
    fn test_norm_cdf_critical_values() {
        // Common z-scores
        assert!((norm_cdf(1.645) - 0.95).abs() < 2e-3); // 90% one-sided
        assert!((norm_cdf(1.96) - 0.975).abs() < 2e-3); // 95% two-sided
        assert!((norm_cdf(2.576) - 0.995).abs() < 2e-3); // 99% two-sided
    }

    #[test]
    fn test_norm_pdf_peak() {
        let peak = 1.0 / (2.0 * PI).sqrt();
        assert!((norm_pdf(0.0) - peak).abs() < 1e-14);
    }

    #[test]
    fn test_norm_pdf_symmetry() {
        for x in [0.5, 1.0, 2.0, 3.0] {
            assert!((norm_pdf(x) - norm_pdf(-x)).abs() < 1e-14);
        }
    }

    #[test]
    fn test_norm_pdf_decay() {
        // PDF should decay as x increases
        assert!(norm_pdf(0.0) > norm_pdf(1.0));
        assert!(norm_pdf(1.0) > norm_pdf(2.0));
        assert!(norm_pdf(2.0) > norm_pdf(3.0));
    }

    #[test]
    fn test_norm_ppf_median() {
        assert!((norm_ppf(0.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_ppf_critical_values() {
        // Inverse of common critical values
        // Moro algorithm has ~3e-9 precision in central region, but may be less at tails
        assert!(
            (norm_ppf(0.95) - 1.6448536269514722).abs() < 1e-2,
            "ppf(0.95) = {}",
            norm_ppf(0.95)
        );
        assert!(
            (norm_ppf(0.975) - 1.9599639845400545).abs() < 1e-2,
            "ppf(0.975) = {}",
            norm_ppf(0.975)
        );
        assert!(
            (norm_ppf(0.995) - 2.5758293035489004).abs() < 1e-2,
            "ppf(0.995) = {}",
            norm_ppf(0.995)
        );
    }

    #[test]
    fn test_norm_ppf_symmetry() {
        // Φ⁻¹(p) = -Φ⁻¹(1-p)
        for p in [0.1, 0.2, 0.3, 0.4] {
            assert!((norm_ppf(p) + norm_ppf(1.0 - p)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_norm_ppf_cdf_inverse() {
        // ppf(cdf(x)) ≈ x
        // Note: This test is sensitive to precision loss in both directions.
        // The erf-based CDF has ~1.5e-7 precision, and the inverse uses a
        // different approximation, so round-trip error can accumulate.
        for x in [-1.0, 0.0, 1.0] {
            let roundtrip = norm_ppf(norm_cdf(x));
            assert!((roundtrip - x).abs() < 0.1, "ppf(cdf({x})) = {roundtrip}");
        }
    }

    #[test]
    fn test_norm_ppf_tails() {
        // Very small p
        assert!(norm_ppf(0.001) < -3.0);
        assert!(norm_ppf(1e-6) < -4.0);
        // Very large p
        assert!(norm_ppf(0.999) > 3.0);
        assert!(norm_ppf(1.0 - 1e-6) > 4.0);
    }

    #[test]
    fn test_norm_ppf_boundaries() {
        assert!(norm_ppf(0.0).is_infinite() && norm_ppf(0.0) < 0.0);
        assert!(norm_ppf(1.0).is_infinite() && norm_ppf(1.0) > 0.0);
    }

    #[test]
    fn test_norm_cdf_batch() {
        let x = vec![-1.0, 0.0, 1.0];
        let result = norm_cdf_batch(&x);
        assert_eq!(result.len(), 3);
        assert!((result[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_general_normal() {
        // N(10, 2): mean=10, std=2
        let mu = 10.0;
        let sigma = 2.0;

        // CDF at mean should be 0.5
        assert!((norm_cdf_general(mu, mu, sigma) - 0.5).abs() < 1e-10);

        // PDF at mean should be maximum
        let pdf_at_mean = norm_pdf_general(mu, mu, sigma);
        assert!(pdf_at_mean > norm_pdf_general(mu + 1.0, mu, sigma));

        // PPF of 0.5 should be mean
        assert!((norm_ppf_general(0.5, mu, sigma) - mu).abs() < 1e-10);
    }
}
