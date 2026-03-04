// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chi-Squared Distribution
//!
//! The chi-squared distribution with k degrees of freedom is the distribution
//! of a sum of squares of k independent standard normal random variables.
//!
//! # Functions
//!
//! - `chi_squared_pdf(x, k)` - Probability density function
//! - `chi_squared_cdf(x, k)` - Cumulative distribution function
//! - `chi_squared_sf(x, k)` - Survival function (1 - CDF)
//! - `chi_squared_quantile(p, k)` - Inverse CDF (quantile function)
//!
//! # Applications
//!
//! - Goodness-of-fit tests
//! - Independence tests in contingency tables
//! - Variance estimation
//! - Nuclear physics (spectral analysis)
//!
//! # Relation to Gamma Distribution
//!
//! χ²(k) = Gamma(k/2, 2)
//!
//! # References
//!
//! - Numerical Recipes, 3rd Edition, Chapter 6.2
//! - NIST/SEMATECH e-Handbook of Statistical Methods

use crate::error::{BarracudaError, Result};
use crate::special::gamma::{ln_gamma, regularized_gamma_p, regularized_gamma_q};

/// WGSL kernel for chi-squared distribution evaluation (f64).
pub const WGSL_CHI_SQUARED_F64: &str = include_str!("../shaders/special/chi_squared_f64.wgsl");
/// WGSL kernel for decomposed chi-squared test statistic (f64).
pub const WGSL_CHI2_DECOMPOSED_F64: &str =
    include_str!("../shaders/special/chi2_decomposed_f64.wgsl");

/// Chi-squared probability density function
///
/// f(x; k) = x^(k/2-1) e^(-x/2) / (2^(k/2) Γ(k/2))
///
/// # Arguments
///
/// * `x` - Value (x >= 0)
/// * `k` - Degrees of freedom (k > 0, typically integer)
///
/// # Example
///
/// ```
/// use barracuda::special::chi_squared_pdf;
///
/// let pdf = chi_squared_pdf(2.0, 3.0)?;
/// assert!(pdf > 0.0);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn chi_squared_pdf(x: f64, k: f64) -> Result<f64> {
    if k <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_pdf requires k > 0, got {k}"),
        });
    }
    if x < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_pdf requires x >= 0, got {x}"),
        });
    }

    if x == 0.0 {
        if k < 2.0 {
            return Ok(f64::INFINITY);
        } else if (k - 2.0).abs() < 1e-10 {
            return Ok(0.5);
        } else {
            return Ok(0.0);
        }
    }

    let half_k = k / 2.0;
    let ln_pdf = (half_k - 1.0) * x.ln() - x / 2.0 - half_k * 2.0_f64.ln() - ln_gamma(half_k)?;

    Ok(ln_pdf.exp())
}

/// Chi-squared cumulative distribution function
///
/// P(X ≤ x) = P(k/2, x/2) where P is the regularized lower incomplete gamma
///
/// # Arguments
///
/// * `x` - Value (x >= 0)
/// * `k` - Degrees of freedom (k > 0)
///
/// # Returns
///
/// Probability that X ≤ x
///
/// # Example
///
/// ```
/// use barracuda::special::chi_squared_cdf;
///
/// // Critical value for k=1 at p=0.95 is approximately 3.841
/// let p = chi_squared_cdf(3.841, 1.0)?;
/// assert!((p - 0.95).abs() < 0.01);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn chi_squared_cdf(x: f64, k: f64) -> Result<f64> {
    if k <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_cdf requires k > 0, got {k}"),
        });
    }
    if x < 0.0 {
        return Ok(0.0);
    }

    // χ²(k) CDF = P(k/2, x/2)
    regularized_gamma_p(k / 2.0, x / 2.0)
}

/// Chi-squared survival function (1 - CDF)
///
/// P(X > x) = Q(k/2, x/2) where Q is the regularized upper incomplete gamma
///
/// # Arguments
///
/// * `x` - Value (x >= 0)
/// * `k` - Degrees of freedom (k > 0)
///
/// # Returns
///
/// Probability that X > x (p-value for chi-squared test)
pub fn chi_squared_sf(x: f64, k: f64) -> Result<f64> {
    if k <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_sf requires k > 0, got {k}"),
        });
    }
    if x < 0.0 {
        return Ok(1.0);
    }

    regularized_gamma_q(k / 2.0, x / 2.0)
}

/// Chi-squared quantile function (inverse CDF)
///
/// Returns x such that P(X ≤ x) = p
///
/// Uses Newton-Raphson iteration starting from an initial guess.
///
/// # Arguments
///
/// * `p` - Probability (0 < p < 1)
/// * `k` - Degrees of freedom (k > 0)
///
/// # Returns
///
/// The quantile (critical value)
///
/// # Example
///
/// ```
/// use barracuda::special::chi_squared_quantile;
///
/// // 95th percentile for k=2 (exponential distribution scaled by 2)
/// let x = chi_squared_quantile(0.95, 2.0)?;
/// assert!((x - 5.9915).abs() < 0.01);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn chi_squared_quantile(p: f64, k: f64) -> Result<f64> {
    if k <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_quantile requires k > 0, got {k}"),
        });
    }
    if p <= 0.0 || p >= 1.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("chi_squared_quantile requires 0 < p < 1, got {p}"),
        });
    }

    // Use bisection method for robustness
    // Find bracket [a, b] such that CDF(a) < p < CDF(b)
    let mut a = 0.001;
    let mut b = k.max(1.0);

    // Expand upper bound until CDF(b) > p
    for _ in 0..50 {
        let cdf_b = chi_squared_cdf(b, k)?;
        if cdf_b > p {
            break;
        }
        b *= 2.0;
    }

    // Bisection search
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    for _ in 0..MAX_ITER {
        let mid = 0.5 * (a + b);

        if (b - a) < TOL * mid {
            return Ok(mid);
        }

        let cdf_mid = chi_squared_cdf(mid, k)?;

        if cdf_mid < p {
            a = mid;
        } else {
            b = mid;
        }
    }

    Ok(0.5 * (a + b))
}

/// Mean of chi-squared distribution
pub fn chi_squared_mean(k: f64) -> f64 {
    k
}

/// Variance of chi-squared distribution
pub fn chi_squared_variance(k: f64) -> f64 {
    2.0 * k
}

/// Mode of chi-squared distribution (for k >= 2)
pub fn chi_squared_mode(k: f64) -> f64 {
    (k - 2.0).max(0.0)
}

/// Chi-squared test statistic
///
/// Computes χ² = Σ (observed - expected)² / expected
///
/// # Arguments
///
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies
///
/// # Returns
///
/// Chi-squared statistic
///
/// # Example
///
/// ```
/// use barracuda::special::chi_squared_statistic;
///
/// let observed = vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
/// let expected = vec![14.67, 14.67, 14.67, 14.67, 14.67, 14.67];
/// let chi2 = chi_squared_statistic(&observed, &expected)?;
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
/// Chi-squared statistic (alias: absorption L-003).
///
/// Same as `chi_squared_statistic`; provided for primal compatibility.
pub fn chi_squared_f64(observed: &[f64], expected: &[f64]) -> Result<f64> {
    chi_squared_statistic(observed, expected)
}

pub fn chi_squared_statistic(observed: &[f64], expected: &[f64]) -> Result<f64> {
    if observed.len() != expected.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "observed and expected must have same length: {} vs {}",
                observed.len(),
                expected.len()
            ),
        });
    }

    let mut chi2 = 0.0;
    for (o, e) in observed.iter().zip(expected.iter()) {
        if *e <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "expected frequencies must be positive".to_string(),
            });
        }
        chi2 += (o - e).powi(2) / e;
    }

    Ok(chi2)
}

/// Perform a chi-squared goodness-of-fit test
///
/// # Arguments
///
/// * `observed` - Observed frequencies
/// * `expected` - Expected frequencies
///
/// # Returns
///
/// (chi2_statistic, p_value, degrees_of_freedom)
///
/// # Example
///
/// ```
/// use barracuda::special::chi_squared_test;
///
/// // Fair die test: 60 rolls, expect 10 of each
/// let observed = vec![8.0, 12.0, 11.0, 9.0, 10.0, 10.0];
/// let expected = vec![10.0; 6];
///
/// let (chi2, p_value, df) = chi_squared_test(&observed, &expected)?;
/// assert_eq!(df, 5); // 6 - 1 = 5 degrees of freedom
/// assert!(p_value > 0.05); // Likely fair die
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> Result<(f64, f64, usize)> {
    let chi2 = chi_squared_statistic(observed, expected)?;
    let df = observed.len() - 1;
    let p_value = chi_squared_sf(chi2, df as f64)?;

    Ok((chi2, p_value, df))
}

// ── Batched chi-squared PDF/CDF GPU dispatch ─────────────────────────────────

/// GPU executor for batched chi-squared PDF + CDF evaluation.
///
/// Evaluates `chi2_pdf(x, k)` and `chi2_cdf(x, k)` for an array of x-values
/// at fixed degrees of freedom `k`. Single dispatch produces both PDF and CDF.
#[cfg(feature = "gpu")]
pub struct ChiSquaredBatchGpu {
    device: std::sync::Arc<crate::device::WgpuDevice>,
}

/// Results from batched chi-squared evaluation.
#[derive(Debug, Clone)]
pub struct ChiSquaredBatchResult {
    pub pdf: Vec<f64>,
    pub cdf: Vec<f64>,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Chi2GpuParams {
    size: u32,
    df: u32,
}

#[cfg(feature = "gpu")]
impl ChiSquaredBatchGpu {
    pub fn new(device: std::sync::Arc<crate::device::WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Evaluate chi-squared PDF and CDF for `x_values` at `df` degrees of freedom.
    pub fn dispatch(&self, x_values: &[f64], df: u32) -> Result<ChiSquaredBatchResult> {
        use crate::device::compute_pipeline::ComputeDispatch;

        let n = x_values.len();
        let input_buf = self.device.create_buffer_f64_init("chi2:input", x_values);
        let pdf_buf = self.device.create_buffer_f64(n)?;
        let cdf_buf = self.device.create_buffer_f64(n)?;
        let params = Chi2GpuParams { size: n as u32, df };
        let params_buf = self.device.create_uniform_buffer("chi2:params", &params);

        let wg = (n as u32).div_ceil(256);
        ComputeDispatch::new(&self.device, "chi_squared_f64")
            .shader(WGSL_CHI_SQUARED_F64, "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &pdf_buf)
            .uniform(2, &params_buf)
            .storage_rw(3, &cdf_buf)
            .dispatch(wg, 1, 1)
            .submit();

        let pdf = self.device.read_f64_buffer(&pdf_buf, n)?;
        let cdf = self.device.read_f64_buffer(&cdf_buf, n)?;
        Ok(ChiSquaredBatchResult { pdf, cdf })
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_pdf_k1() {
        // For k=1, PDF at x=1 should be about 0.24
        let pdf = chi_squared_pdf(1.0, 1.0).unwrap();
        assert!((pdf - 0.2420).abs() < 0.01);
    }

    #[test]
    fn test_chi_squared_pdf_k2() {
        // For k=2, PDF is exponential: f(x) = e^(-x/2)/2
        let pdf = chi_squared_pdf(2.0, 2.0).unwrap();
        let expected = (-1.0_f64).exp() / 2.0;
        assert!((pdf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_cdf_k2() {
        // For k=2, CDF is 1 - e^(-x/2)
        let cdf = chi_squared_cdf(2.0, 2.0).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((cdf - expected).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_cdf_critical_values() {
        // Well-known critical values
        // k=1, p=0.95: x ≈ 3.841
        let cdf = chi_squared_cdf(3.841, 1.0).unwrap();
        assert!((cdf - 0.95).abs() < 0.01);

        // k=2, p=0.95: x ≈ 5.991
        let cdf = chi_squared_cdf(5.991, 2.0).unwrap();
        assert!((cdf - 0.95).abs() < 0.01);

        // k=10, p=0.95: x ≈ 18.31
        let cdf = chi_squared_cdf(18.31, 10.0).unwrap();
        assert!((cdf - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_chi_squared_sf_complement() {
        let x = 5.0;
        let k = 3.0;
        let cdf = chi_squared_cdf(x, k).unwrap();
        let sf = chi_squared_sf(x, k).unwrap();

        assert!((cdf + sf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_quantile_inverse_cdf() {
        // Test for various k and p values
        for k in [2.0, 5.0, 10.0] {
            for p in [0.5, 0.9, 0.95] {
                let x = chi_squared_quantile(p, k).unwrap();
                let p_check = chi_squared_cdf(x, k).unwrap();

                assert!(
                    (p - p_check).abs() < 1e-4,
                    "Failed for k={}, p={}: got p_check={}",
                    k,
                    p,
                    p_check
                );
            }
        }
    }

    #[test]
    fn test_chi_squared_quantile_known_values() {
        // k=1, p=0.95 -> x ≈ 3.841
        let x = chi_squared_quantile(0.95, 1.0).unwrap();
        assert!((x - 3.841).abs() < 0.01);

        // k=2, p=0.95 -> x ≈ 5.991
        let x = chi_squared_quantile(0.95, 2.0).unwrap();
        assert!((x - 5.991).abs() < 0.01);
    }

    #[test]
    fn test_chi_squared_moments() {
        let k = 5.0;

        assert_eq!(chi_squared_mean(k), 5.0);
        assert_eq!(chi_squared_variance(k), 10.0);
        assert_eq!(chi_squared_mode(k), 3.0);
    }

    #[test]
    fn test_chi_squared_f64_alias() {
        // L-003: chi_squared_f64 matches chi_squared_statistic
        let observed = vec![16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
        let expected = vec![14.67, 14.67, 14.67, 14.67, 14.67, 14.67];
        let via_f64 = chi_squared_f64(&observed, &expected).unwrap();
        let via_stat = chi_squared_statistic(&observed, &expected).unwrap();
        assert!((via_f64 - via_stat).abs() < 1e-14);
        // Known: fair die (obs=exp) gives chi²=0
        let fair = chi_squared_f64(&[10.0; 6], &[10.0; 6]).unwrap();
        assert!((fair - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_chi_squared_statistic() {
        // Fair die test
        let observed = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let expected = vec![10.0; 6];

        let chi2 = chi_squared_statistic(&observed, &expected).unwrap();
        assert!((chi2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi_squared_test_fair_die() {
        // Nearly perfect observations
        let observed = vec![10.0, 9.0, 11.0, 10.0, 10.0, 10.0];
        let expected = vec![10.0; 6];

        let (chi2, p_value, df) = chi_squared_test(&observed, &expected).unwrap();

        assert_eq!(df, 5);
        assert!(chi2 < 1.0); // Small chi2 for good fit
        assert!(p_value > 0.9); // High p-value means data fits expected
    }

    #[test]
    fn test_chi_squared_test_unfair_die() {
        // Biased towards 6
        let observed = vec![5.0, 5.0, 5.0, 5.0, 5.0, 35.0];
        let expected = vec![10.0; 6];

        let (chi2, p_value, df) = chi_squared_test(&observed, &expected).unwrap();

        assert_eq!(df, 5);
        assert!(chi2 > 50.0); // Large chi2 for poor fit
        assert!(p_value < 0.01); // Low p-value indicates significant deviation
    }

    #[test]
    fn test_chi_squared_invalid_inputs() {
        assert!(chi_squared_pdf(-1.0, 2.0).is_err());
        assert!(chi_squared_cdf(1.0, 0.0).is_err());
        assert!(chi_squared_quantile(0.0, 2.0).is_err());
        assert!(chi_squared_quantile(1.0, 2.0).is_err());
    }
}
