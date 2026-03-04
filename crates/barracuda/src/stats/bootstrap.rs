// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bootstrap confidence intervals
//!
//! Provides non-parametric confidence interval estimation using the bootstrap
//! resampling method. Works for any statistic that can be computed from a sample.
//!
//! # Algorithm
//!
//! 1. Resample data with replacement B times
//! 2. Compute statistic on each bootstrap sample
//! 3. Use percentiles of bootstrap distribution for CI
//!
//! # Reference
//!
//! - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
//! - hotSpring validation: `stats.rs::bootstrap_ci()`

use crate::error::{BarracudaError, Result};

#[cfg(feature = "gpu")]
use crate::device::capabilities::WORKGROUP_SIZE_1D;
#[cfg(feature = "gpu")]
use crate::device::compute_pipeline::ComputeDispatch;

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapCI {
    /// Point estimate (statistic on original data)
    pub estimate: f64,
    /// Lower bound of confidence interval
    pub lower: f64,
    /// Upper bound of confidence interval
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence: f64,
    /// Standard error (std dev of bootstrap distribution)
    pub std_error: f64,
    /// Number of bootstrap resamples
    pub n_bootstrap: usize,
    /// Bootstrap distribution (sorted)
    pub distribution: Vec<f64>,
}

impl BootstrapCI {
    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let pct = (self.confidence * 100.0).round() as u32;
        format!(
            "{:.4} ({}% CI: [{:.4}, {:.4}], SE: {:.4})",
            self.estimate, pct, self.lower, self.upper, self.std_error
        )
    }
}

/// Simple LCG random number generator for bootstrap sampling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters (Numerical Recipes)
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn usize_range(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Compute bootstrap confidence interval for any statistic.
///
/// # Arguments
///
/// * `data` - Sample data
/// * `statistic` - Function computing the statistic from a sample
/// * `n_bootstrap` - Number of bootstrap resamples (default: 1000)
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// [`BootstrapCI`] with point estimate, confidence bounds, and standard error.
///
/// # Example
///
/// ```
/// use barracuda::stats::bootstrap_ci;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
///
/// // Bootstrap CI for mean
/// let ci = bootstrap_ci(
///     &data,
///     |sample| sample.iter().sum::<f64>() / sample.len() as f64,
///     1000,
///     0.95,
///     42,
/// ).unwrap();
///
/// println!("Mean: {}", ci.summary());
/// assert!(ci.lower < 5.5 && ci.upper > 5.5);  // True mean is 5.5
/// ```
pub fn bootstrap_ci<F>(
    data: &[f64],
    statistic: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapCI>
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "data cannot be empty".to_string(),
        });
    }
    if n_bootstrap == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "n_bootstrap must be > 0".to_string(),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(BarracudaError::InvalidInput {
            message: format!("confidence must be in (0, 1), got {confidence}"),
        });
    }

    // Point estimate on original data
    let estimate = statistic(data);

    // Bootstrap resampling
    let mut rng = SimpleRng::new(seed);
    let mut bootstrap_stats = Vec::with_capacity(n_bootstrap);
    let mut resample = vec![0.0; n];

    for _ in 0..n_bootstrap {
        // Resample with replacement
        for i in 0..n {
            resample[i] = data[rng.usize_range(n)];
        }
        bootstrap_stats.push(statistic(&resample));
    }

    // Sort for percentile computation
    bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Percentile CI
    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;
    let upper_idx = upper_idx.min(n_bootstrap - 1);

    let lower = bootstrap_stats[lower_idx];
    let upper = bootstrap_stats[upper_idx];

    // Standard error
    let mean: f64 = bootstrap_stats.iter().sum::<f64>() / n_bootstrap as f64;
    let variance: f64 = bootstrap_stats
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (n_bootstrap - 1) as f64;
    let std_error = variance.sqrt();

    Ok(BootstrapCI {
        estimate,
        lower,
        upper,
        confidence,
        std_error,
        n_bootstrap,
        distribution: bootstrap_stats,
    })
}

/// Bootstrap CI for the mean.
///
/// Convenience function for the common case of bootstrapping the mean.
///
/// # Example
///
/// ```
/// use barracuda::stats::bootstrap_mean;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let ci = bootstrap_mean(&data, 1000, 0.95, 42).unwrap();
/// println!("Mean: {}", ci.summary());
/// ```
pub fn bootstrap_mean(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapCI> {
    bootstrap_ci(
        data,
        |sample| sample.iter().sum::<f64>() / sample.len() as f64,
        n_bootstrap,
        confidence,
        seed,
    )
}

/// Bootstrap CI for the median.
///
/// # Example
///
/// ```
/// use barracuda::stats::bootstrap_median;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];  // Outlier
/// let ci = bootstrap_median(&data, 1000, 0.95, 42).unwrap();
/// println!("Median: {}", ci.summary());  // Robust to outlier
/// ```
pub fn bootstrap_median(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapCI> {
    bootstrap_ci(
        data,
        |sample| {
            let mut sorted = sample.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            }
        },
        n_bootstrap,
        confidence,
        seed,
    )
}

/// RAWR — Resampling with Analytical Weights for Reproducibility.
///
/// Implements Dirichlet-weighted mean resampling (Wang et al. 2021,
/// Bioinformatics/ISMB). Each replicate draws n independent Exp(1) variates,
/// normalizes them to Dirichlet weights, and computes a weighted mean.
/// This is smoother than percentile bootstrap and particularly effective
/// for small samples common in ecology.
///
/// # Provenance
///
/// Absorbed from groundSpring `bootstrap.rs::rawr_mean()` (V7).
pub fn rawr_mean(
    data: &[f64],
    n_replicates: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapCI> {
    let n = data.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "data cannot be empty".to_string(),
        });
    }
    if n_replicates == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "n_replicates must be > 0".to_string(),
        });
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(BarracudaError::InvalidInput {
            message: format!("confidence must be in (0, 1), got {confidence}"),
        });
    }

    let estimate = data.iter().sum::<f64>() / n as f64;
    let mut rng = SimpleRng::new(seed);
    let mut means = Vec::with_capacity(n_replicates);

    const EXP_CAP: f64 = 30.0;
    let mut weights = vec![0.0f64; n];

    for _ in 0..n_replicates {
        let mut wsum = 0.0;
        for w in weights.iter_mut() {
            let u = rng.next_f64();
            *w = if u > 0.0 { -u.ln() } else { EXP_CAP };
            wsum += *w;
        }

        let mut weighted_mean = 0.0;
        for (j, &d) in data.iter().enumerate() {
            weighted_mean = (weights[j] / wsum).mul_add(d, weighted_mean);
        }
        means.push(weighted_mean);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_replicates as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_replicates as f64)
        .ceil()
        .min((n_replicates - 1) as f64) as usize;

    let mean_of_means: f64 = means.iter().sum::<f64>() / n_replicates as f64;
    let variance: f64 = means
        .iter()
        .map(|x| (x - mean_of_means).powi(2))
        .sum::<f64>()
        / (n_replicates - 1).max(1) as f64;

    Ok(BootstrapCI {
        estimate,
        lower: means[lower_idx],
        upper: means[upper_idx],
        confidence,
        std_error: variance.sqrt(),
        n_bootstrap: n_replicates,
        distribution: means,
    })
}

/// Bootstrap CI for standard deviation.
///
/// # Example
///
/// ```
/// use barracuda::stats::bootstrap_std;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let ci = bootstrap_std(&data, 1000, 0.95, 42).unwrap();
/// println!("Std: {}", ci.summary());
/// ```
pub fn bootstrap_std(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> Result<BootstrapCI> {
    bootstrap_ci(
        data,
        |sample| {
            let n = sample.len() as f64;
            let mean = sample.iter().sum::<f64>() / n;
            let variance = sample.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        },
        n_bootstrap,
        confidence,
        seed,
    )
}

// ── GPU Bootstrap Mean (ComputeDispatch) ────────────────────────────────────

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BootstrapGpuParams {
    n: u32,
    n_bootstrap: u32,
    seed: u32,
    _pad: u32, // align to 16 bytes for uniform buffer
}

#[cfg(feature = "gpu")]
pub struct BootstrapMeanGpu {
    device: std::sync::Arc<crate::device::WgpuDevice>,
}

#[cfg(feature = "gpu")]
impl BootstrapMeanGpu {
    pub fn new(device: std::sync::Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch GPU-parallel bootstrap mean estimation.
    pub fn dispatch(
        &self,
        data: &[f64],
        n_bootstrap: u32,
        seed: u32,
    ) -> crate::error::Result<Vec<f64>> {
        let n = data.len() as u32;
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "data cannot be empty".to_string(),
            });
        }
        if n_bootstrap == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "n_bootstrap must be > 0".to_string(),
            });
        }

        let params = BootstrapGpuParams {
            n,
            n_bootstrap,
            seed,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("bootstrap_mean:params", &params);

        let data_buf = self
            .device
            .create_buffer_f64_init("bootstrap_mean:data", data);
        let out_buf = self.device.create_buffer_f64(n_bootstrap as usize)?;

        let wg_count = n_bootstrap.div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "bootstrap_mean")
            .shader(super::WGSL_BOOTSTRAP_MEAN_F64, "main")
            .f64()
            .storage_read(0, &data_buf)
            .storage_rw(1, &out_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        self.device.read_f64_buffer(&out_buf, n_bootstrap as usize)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_mean() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let ci = bootstrap_mean(&data, 1000, 0.95, 42).unwrap();

        // True mean is 50.5
        assert!((ci.estimate - 50.5).abs() < 0.01);
        assert!(ci.lower < 50.5 && ci.upper > 50.5);
        assert!(ci.std_error > 0.0);
        assert_eq!(ci.n_bootstrap, 1000);
    }

    #[test]
    fn test_bootstrap_median() {
        let data: Vec<f64> = (1..=99).map(|x| x as f64).collect();
        let ci = bootstrap_median(&data, 1000, 0.95, 42).unwrap();

        // True median is 50.0
        assert!((ci.estimate - 50.0).abs() < 0.01);
        assert!(ci.lower < 50.0 && ci.upper > 50.0);
    }

    #[test]
    fn test_bootstrap_confidence_levels() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();

        let ci_90 = bootstrap_mean(&data, 1000, 0.90, 42).unwrap();
        let ci_95 = bootstrap_mean(&data, 1000, 0.95, 42).unwrap();
        let ci_99 = bootstrap_mean(&data, 1000, 0.99, 42).unwrap();

        // Wider confidence = wider interval
        let width_90 = ci_90.upper - ci_90.lower;
        let width_95 = ci_95.upper - ci_95.lower;
        let width_99 = ci_99.upper - ci_99.lower;

        assert!(width_90 < width_95);
        assert!(width_95 < width_99);
    }

    #[test]
    fn test_bootstrap_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ci = bootstrap_mean(&data, 100, 0.95, 42).unwrap();
        let summary = ci.summary();

        assert!(summary.contains("95%"));
        assert!(summary.contains("CI:"));
        assert!(summary.contains("SE:"));
    }

    #[test]
    fn test_bootstrap_reproducibility() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let ci1 = bootstrap_mean(&data, 100, 0.95, 42).unwrap();
        let ci2 = bootstrap_mean(&data, 100, 0.95, 42).unwrap();

        // Same seed = same result
        assert!((ci1.lower - ci2.lower).abs() < 1e-10);
        assert!((ci1.upper - ci2.upper).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_custom_statistic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Bootstrap for max
        let ci = bootstrap_ci(
            &data,
            |sample| sample.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            1000,
            0.95,
            42,
        )
        .unwrap();

        assert!((ci.estimate - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_errors() {
        assert!(bootstrap_mean(&[], 100, 0.95, 42).is_err());
        assert!(bootstrap_mean(&[1.0, 2.0], 0, 0.95, 42).is_err());
        assert!(bootstrap_mean(&[1.0, 2.0], 100, 1.5, 42).is_err());
        assert!(bootstrap_mean(&[1.0, 2.0], 100, -0.1, 42).is_err());
    }

    #[test]
    fn test_rawr_mean_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = rawr_mean(&data, 2000, 0.95, 42).unwrap();
        assert!(
            (ci.estimate - 5.5).abs() < 1e-10,
            "point estimate should be arithmetic mean"
        );
        assert!(
            ci.lower < 5.5 && ci.upper > 5.5,
            "CI should bracket true mean"
        );
        assert!(ci.std_error > 0.0);
    }

    #[test]
    fn test_rawr_mean_errors() {
        assert!(rawr_mean(&[], 100, 0.95, 42).is_err());
        assert!(rawr_mean(&[1.0, 2.0], 0, 0.95, 42).is_err());
        assert!(rawr_mean(&[1.0, 2.0], 100, 1.5, 42).is_err());
    }

    #[test]
    fn test_rawr_mean_reproducibility() {
        let data = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let ci1 = rawr_mean(&data, 500, 0.95, 123).unwrap();
        let ci2 = rawr_mean(&data, 500, 0.95, 123).unwrap();
        assert!((ci1.lower - ci2.lower).abs() < 1e-10);
        assert!((ci1.upper - ci2.upper).abs() < 1e-10);
    }

    #[test]
    fn test_rawr_vs_bootstrap_similar_ci_width() {
        let data: Vec<f64> = (1..=50).map(f64::from).collect();
        let boot = bootstrap_mean(&data, 2000, 0.95, 42).unwrap();
        let rawr = rawr_mean(&data, 2000, 0.95, 42).unwrap();
        let boot_width = boot.upper - boot.lower;
        let rawr_width = rawr.upper - rawr.lower;
        assert!(
            (boot_width - rawr_width).abs() / boot_width < 0.5,
            "RAWR and bootstrap CI widths should be in similar ballpark"
        );
    }
}
