// SPDX-License-Identifier: AGPL-3.0-or-later
//! Welford's online algorithm for numerically stable mean, variance, and covariance.
//!
//! Single-pass, constant-memory accumulators for streaming statistics.
//! Absorbed from groundSpring V80 where the Welford co-moment was used
//! for numerically stable covariance in long time series.
//!
//! # Advantages over two-pass algorithms
//!
//! - **Numerically stable**: avoids catastrophic cancellation that plagues
//!   `Σ(x - x̄)²` when the mean is large relative to the variance.
//! - **Single pass**: processes data in O(n) time with O(1) memory.
//! - **Streaming**: can incorporate new data points without reprocessing.
//! - **Mergeable**: two `WelfordState` accumulators can be combined
//!   (Chan's parallel algorithm).
//!
//! # Usage
//!
//! ```
//! use barracuda::stats::welford::{WelfordState, WelfordCovState};
//!
//! // Online mean and variance
//! let mut state = WelfordState::new();
//! for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
//!     state.update(x);
//! }
//! assert!((state.mean() - 5.0).abs() < 1e-10);
//! assert!((state.population_variance() - 4.0).abs() < 1e-10);
//!
//! // Online covariance
//! let mut cov = WelfordCovState::new();
//! let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
//! for (&x, &y) in xs.iter().zip(ys.iter()) {
//!     cov.update(x, y);
//! }
//! assert!((cov.sample_covariance() - 5.0).abs() < 1e-10);
//! ```
//!
//! # References
//!
//! - Welford (1962), Technometrics 4(3):419-420
//! - Knuth TAOCP Vol 2, §4.2.2
//! - Chan, Golub & `LeVeque` (1979), parallel variance

/// Online accumulator for mean and variance using Welford's algorithm.
///
/// Uses the recurrence:
/// ```text
/// δ  = xₙ - μₙ₋₁
/// μₙ = μₙ₋₁ + δ/n
/// M₂ₙ = M₂ₙ₋₁ + δ·(xₙ - μₙ)
/// ```
#[derive(Debug, Clone)]
pub struct WelfordState {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    /// Create a new empty accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Incorporate a new data point.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Number of data points incorporated.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Current running mean.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (divide by n).
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count < 1 {
            return 0.0;
        }
        self.m2 / self.count as f64
    }

    /// Sample variance (divide by n-1).
    #[must_use]
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Merge another `WelfordState` into this one (Chan's parallel algorithm).
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = delta.mul_add(other.count as f64 / combined_count as f64, self.mean);
        let combined_m2 = (delta * delta).mul_add(
            self.count as f64 * other.count as f64 / combined_count as f64,
            self.m2 + other.m2,
        );
        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
    }

    /// Build from a complete slice (convenience for non-streaming use).
    #[must_use]
    pub fn from_slice(data: &[f64]) -> Self {
        let mut state = Self::new();
        for &x in data {
            state.update(x);
        }
        state
    }
}

impl Default for WelfordState {
    fn default() -> Self {
        Self::new()
    }
}

/// Online accumulator for bivariate mean, variance, and co-moment
/// using Welford's algorithm extended to two variables.
///
/// The co-moment `C₁₂` tracks `Σ(xᵢ - μₓ)(yᵢ - μᵧ)` incrementally:
/// ```text
/// δₓ = xₙ - μₓ,ₙ₋₁
/// μₓ,ₙ = μₓ,ₙ₋₁ + δₓ/n
/// δᵧ = yₙ - μᵧ,ₙ₋₁
/// μᵧ,ₙ = μᵧ,ₙ₋₁ + δᵧ/n
/// C₁₂,ₙ = C₁₂,ₙ₋₁ + δₓ·(yₙ - μᵧ,ₙ)
/// ```
#[derive(Debug, Clone)]
pub struct WelfordCovState {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    co_moment: f64,
}

impl WelfordCovState {
    /// Create a new empty bivariate accumulator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            m2_x: 0.0,
            m2_y: 0.0,
            co_moment: 0.0,
        }
    }

    /// Incorporate a new (x, y) data pair.
    pub fn update(&mut self, x: f64, y: f64) {
        self.count += 1;
        let n = self.count as f64;
        let dx = x - self.mean_x;
        self.mean_x += dx / n;
        let dy = y - self.mean_y;
        self.mean_y += dy / n;
        // co-moment uses the *updated* mean_y but *old* delta_x
        self.co_moment += dx * (y - self.mean_y);
        // individual M2 updates (standard Welford)
        self.m2_x += dx * (x - self.mean_x);
        self.m2_y += dy * (y - self.mean_y);
    }

    /// Number of data pairs incorporated.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Running mean of x.
    #[must_use]
    pub const fn mean_x(&self) -> f64 {
        self.mean_x
    }

    /// Running mean of y.
    #[must_use]
    pub const fn mean_y(&self) -> f64 {
        self.mean_y
    }

    /// Population covariance (divide by n).
    #[must_use]
    pub fn population_covariance(&self) -> f64 {
        if self.count < 1 {
            return 0.0;
        }
        self.co_moment / self.count as f64
    }

    /// Sample covariance (divide by n-1).
    #[must_use]
    pub fn sample_covariance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.co_moment / (self.count - 1) as f64
    }

    /// Pearson correlation coefficient from running accumulators.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom == 0.0 {
            return f64::NAN;
        }
        self.co_moment / denom
    }

    /// Sample variance of x.
    #[must_use]
    pub fn variance_x(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2_x / (self.count - 1) as f64
    }

    /// Sample variance of y.
    #[must_use]
    pub fn variance_y(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2_y / (self.count - 1) as f64
    }

    /// Merge another state (Chan's parallel algorithm for co-moment).
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let combined = self.count + other.count;
        let dx = other.mean_x - self.mean_x;
        let dy = other.mean_y - self.mean_y;
        let factor = self.count as f64 * other.count as f64 / combined as f64;
        self.co_moment += (dx * dy).mul_add(factor, other.co_moment);
        self.m2_x += (dx * dx).mul_add(factor, other.m2_x);
        self.m2_y += (dy * dy).mul_add(factor, other.m2_y);
        self.mean_x += dx * (other.count as f64 / combined as f64);
        self.mean_y += dy * (other.count as f64 / combined as f64);
        self.count = combined;
    }

    /// Build from paired slices (convenience for non-streaming use).
    ///
    /// If lengths differ, processes only the shorter length.
    #[must_use]
    pub fn from_slices(x: &[f64], y: &[f64]) -> Self {
        let mut state = Self::new();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            state.update(xi, yi);
        }
        state
    }
}

impl Default for WelfordCovState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_mean() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let state = WelfordState::from_slice(&data);
        assert!((state.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn welford_population_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let state = WelfordState::from_slice(&data);
        assert!((state.population_variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn welford_sample_variance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let state = WelfordState::from_slice(&data);
        assert!((state.sample_variance() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn welford_empty() {
        let state = WelfordState::new();
        assert_eq!(state.count(), 0);
        assert_eq!(state.population_variance(), 0.0);
        assert_eq!(state.sample_variance(), 0.0);
    }

    #[test]
    fn welford_single_point() {
        let mut state = WelfordState::new();
        state.update(42.0);
        assert!((state.mean() - 42.0).abs() < 1e-15);
        assert_eq!(state.population_variance(), 0.0);
        assert_eq!(state.sample_variance(), 0.0);
    }

    #[test]
    fn welford_merge() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let full = WelfordState::from_slice(&data);

        let mut a = WelfordState::from_slice(&data[..4]);
        let b = WelfordState::from_slice(&data[4..]);
        a.merge(&b);

        assert!((a.mean() - full.mean()).abs() < 1e-10);
        assert!((a.sample_variance() - full.sample_variance()).abs() < 1e-10);
    }

    #[test]
    fn welford_cov_perfect_positive() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0];
        let state = WelfordCovState::from_slices(&xs, &ys);
        assert!((state.sample_covariance() - 5.0).abs() < 1e-10);
        assert!((state.correlation() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn welford_cov_perfect_negative() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [10.0, 8.0, 6.0, 4.0, 2.0];
        let state = WelfordCovState::from_slices(&xs, &ys);
        assert!((state.sample_covariance() + 5.0).abs() < 1e-10);
        assert!((state.correlation() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn welford_cov_self_is_variance() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let state = WelfordCovState::from_slices(&xs, &xs);
        assert!((state.sample_covariance() - 2.5).abs() < 1e-10);
        assert!((state.variance_x() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn welford_cov_merge() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ys = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let full = WelfordCovState::from_slices(&xs, &ys);

        let mut a = WelfordCovState::from_slices(&xs[..4], &ys[..4]);
        let b = WelfordCovState::from_slices(&xs[4..], &ys[4..]);
        a.merge(&b);

        assert!((a.sample_covariance() - full.sample_covariance()).abs() < 1e-10);
        assert!((a.correlation() - full.correlation()).abs() < 1e-10);
    }

    #[test]
    fn welford_numerical_stability_large_mean() {
        // Classic catastrophic cancellation case:
        // x values near 1e9, variance is small
        let data: Vec<f64> = (0..1000).map(|i| 1e9 + i as f64).collect();
        let state = WelfordState::from_slice(&data);

        let expected_mean = 1e9 + 499.5;
        assert!((state.mean() - expected_mean).abs() < 1e-6);

        // Variance of 0..999 = (999*1000*(2*999+1))/6 / 1000 (pop) - 499.5^2
        // = 83_333.25 (population variance)
        let expected_var = 83_333.25;
        assert!(
            (state.population_variance() - expected_var).abs() < 0.01,
            "got {}",
            state.population_variance()
        );
    }
}
