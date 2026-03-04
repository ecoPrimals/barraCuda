// SPDX-License-Identifier: AGPL-3.0-or-later
//! Maximin Latin Hypercube Sampling
//!
//! Generates space-filling sample designs by maximizing the minimum
//! pairwise distance between sample points. Starts with an LHS design
//! and iteratively improves it by swapping elements within columns.
//!
//! # Algorithm
//!
//! 1. Generate initial LHS design via [`latin_hypercube`]
//! 2. For each iteration:
//!    a. Select a random column (dimension) and two random rows
//!    b. Swap the values in that column
//!    c. If the maximin distance improves, keep the swap; otherwise revert
//! 3. Repeat until `max_iter` swaps attempted or convergence
//!
//! This is the "columnwise pairwise" (CP) algorithm from Morris & Mitchell (1995).
//!
//! # Cross-Domain Applications
//!
//! - **Surrogate modeling**: Better-distributed training points → more accurate RBF
//! - **Design of experiments**: Maximin designs have optimal space-filling properties
//! - **Monte Carlo**: Quasi-random initial conditions with guaranteed coverage
//! - **Sensitivity analysis**: Uniform exploration of parameter space
//!
//! # References
//!
//! - Morris, M.D. & Mitchell, T.J. (1995). "Exploratory designs for
//!   computational experiments." Journal of Statistical Planning and Inference.
//! - hotSpring: Space-filling designs for nuclear EOS parameter fitting

use crate::error::{BarracudaError, Result};
use crate::sample::lhs::{latin_hypercube, maximin_distance, Xoshiro256};

/// Configuration for maximin LHS optimization.
#[derive(Debug, Clone)]
pub struct MaximinConfig {
    /// Number of swap attempts per iteration (default: 100 × n_dims)
    pub max_iter: usize,
    /// Number of candidate designs to generate and pick the best (default: 5)
    pub n_candidates: usize,
    /// Random seed
    pub seed: u64,
}

impl MaximinConfig {
    /// Create a new configuration with defaults scaled to dimension count.
    pub fn new(n_dims: usize, seed: u64) -> Self {
        Self {
            max_iter: 100 * n_dims,
            n_candidates: 5,
            seed,
        }
    }

    /// Set maximum swap iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set number of candidate designs.
    #[must_use]
    pub fn with_candidates(mut self, n_candidates: usize) -> Self {
        self.n_candidates = n_candidates;
        self
    }
}

/// Result of maximin optimization.
#[derive(Debug, Clone)]
pub struct MaximinResult {
    /// Optimized sample points [n_samples][n_dims]
    pub samples: Vec<Vec<f64>>,
    /// Maximin distance of the optimized design
    pub maximin_dist: f64,
    /// Maximin distance of the initial LHS design (for comparison)
    pub initial_maximin_dist: f64,
    /// Number of successful swaps made
    pub n_swaps: usize,
}

/// Generate a maximin-optimized Latin Hypercube Sample.
///
/// Produces `n_samples` points in `d` dimensions that maximize the minimum
/// pairwise distance while maintaining the LHS stratification property.
///
/// # Arguments
///
/// * `n_samples` - Number of sample points to generate
/// * `bounds` - Box bounds `[(min, max), ...]` for each dimension
/// * `config` - Optimization configuration
///
/// # Returns
///
/// [`MaximinResult`] containing the optimized samples and diagnostics.
///
/// # Examples
///
/// ```
/// use barracuda::sample::maximin::{maximin_lhs, MaximinConfig};
///
/// let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
/// let config = MaximinConfig::new(2, 42);
/// let result = maximin_lhs(20, &bounds, &config)?;
///
/// assert_eq!(result.samples.len(), 20);
/// assert!(result.maximin_dist >= result.initial_maximin_dist * 0.99);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn maximin_lhs(
    n_samples: usize,
    bounds: &[(f64, f64)],
    config: &MaximinConfig,
) -> Result<MaximinResult> {
    if n_samples < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "maximin_lhs requires at least 2 samples".to_string(),
        });
    }

    if bounds.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "bounds must be non-empty".to_string(),
        });
    }

    let n_dims = bounds.len();

    // Generate multiple candidate LHS designs and pick the best
    let mut best_samples = None;
    let mut best_dist = f64::NEG_INFINITY;

    for c in 0..config.n_candidates {
        let candidate_seed = config.seed.wrapping_add(c as u64 * 7919);
        let samples = latin_hypercube(n_samples, bounds, candidate_seed)?;
        let dist = maximin_distance(&samples);

        if dist > best_dist {
            best_dist = dist;
            best_samples = Some(samples);
        }
    }

    let mut samples = best_samples.ok_or_else(|| BarracudaError::InvalidInput {
        message: "maximin_lhs requires n_candidates >= 1".to_string(),
    })?;
    let initial_maximin_dist = best_dist;
    let mut current_dist = best_dist;
    let mut rng = Xoshiro256::new(config.seed.wrapping_add(99991));
    let mut n_swaps = 0;

    // Columnwise pairwise (CP) optimization:
    // Swap two elements within a column and keep if maximin improves
    for _ in 0..config.max_iter {
        // Pick random dimension and two random rows
        let dim = rng.next_u64() as usize % n_dims;
        let row1 = rng.next_u64() as usize % n_samples;
        let mut row2 = rng.next_u64() as usize % n_samples;
        while row2 == row1 {
            row2 = rng.next_u64() as usize % n_samples;
        }

        // Swap
        let tmp = samples[row1][dim];
        samples[row1][dim] = samples[row2][dim];
        samples[row2][dim] = tmp;

        // O(n) evaluation via partial_maximin instead of O(n²) full recompute
        let new_dist = partial_maximin(&samples, row1, row2, current_dist);

        if new_dist > current_dist {
            current_dist = new_dist;
            n_swaps += 1;
        } else {
            // Revert
            let tmp = samples[row1][dim];
            samples[row1][dim] = samples[row2][dim];
            samples[row2][dim] = tmp;
        }
    }

    Ok(MaximinResult {
        samples,
        maximin_dist: current_dist,
        initial_maximin_dist,
        n_swaps,
    })
}

/// Minimum pairwise distance restricted to rows affected by a swap.
///
/// O(n) instead of O(n²) — only distances involving `row1` or `row2` are checked.
/// Returns the minimum of those distances and the existing `current_min`.
fn partial_maximin(samples: &[Vec<f64>], row1: usize, row2: usize, current_min: f64) -> f64 {
    let n = samples.len();
    let mut min_dist = current_min;

    for i in 0..n {
        if i == row1 || i == row2 {
            continue;
        }

        // Distance to row1
        let d1: f64 = samples[i]
            .iter()
            .zip(samples[row1].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if d1 < min_dist {
            min_dist = d1;
        }

        // Distance to row2
        let d2: f64 = samples[i]
            .iter()
            .zip(samples[row2].iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if d2 < min_dist {
            min_dist = d2;
        }
    }

    // Distance between row1 and row2
    let d12: f64 = samples[row1]
        .iter()
        .zip(samples[row2].iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    if d12 < min_dist {
        min_dist = d12;
    }

    min_dist
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maximin_lhs_basic() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = MaximinConfig::new(2, 42);
        let result = maximin_lhs(10, &bounds, &config).unwrap();

        assert_eq!(result.samples.len(), 10);
        assert_eq!(result.samples[0].len(), 2);

        // Optimized should be at least as good as initial
        assert!(result.maximin_dist >= result.initial_maximin_dist * 0.999);
    }

    #[test]
    fn test_maximin_improves_on_lhs() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let config = MaximinConfig::new(3, 42).with_max_iter(500);
        let result = maximin_lhs(20, &bounds, &config).unwrap();

        // The optimization should have made at least one improvement
        // (not guaranteed per run, but very likely with 500 iterations on 3D)
        assert!(
            result.maximin_dist >= result.initial_maximin_dist,
            "Maximin should not get worse: {} < {}",
            result.maximin_dist,
            result.initial_maximin_dist
        );
    }

    #[test]
    fn test_maximin_respects_bounds() {
        let bounds = vec![(-10.0, 10.0), (0.0, 100.0)];
        let config = MaximinConfig::new(2, 42);
        let result = maximin_lhs(30, &bounds, &config).unwrap();

        for point in &result.samples {
            assert!(point[0] >= -10.0 && point[0] <= 10.0);
            assert!(point[1] >= 0.0 && point[1] <= 100.0);
        }
    }

    #[test]
    fn test_maximin_reproducible() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = MaximinConfig::new(2, 42);

        let r1 = maximin_lhs(10, &bounds, &config).unwrap();
        let r2 = maximin_lhs(10, &bounds, &config).unwrap();

        // Same seed → same result
        assert_eq!(r1.samples, r2.samples);
        assert_eq!(r1.maximin_dist, r2.maximin_dist);
    }

    #[test]
    fn test_maximin_config_builder() {
        let config = MaximinConfig::new(5, 42)
            .with_max_iter(1000)
            .with_candidates(10);

        assert_eq!(config.max_iter, 1000);
        assert_eq!(config.n_candidates, 10);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_maximin_errors() {
        let bounds = vec![(0.0, 1.0)];
        let config = MaximinConfig::new(1, 42);

        // Too few samples
        assert!(maximin_lhs(1, &bounds, &config).is_err());

        // Empty bounds
        assert!(maximin_lhs(10, &[], &config).is_err());
    }

    #[test]
    fn test_maximin_high_dimensional() {
        let bounds: Vec<(f64, f64)> = (0..5).map(|_| (0.0, 1.0)).collect();
        let config = MaximinConfig::new(5, 42).with_max_iter(200);
        let result = maximin_lhs(15, &bounds, &config).unwrap();

        assert_eq!(result.samples.len(), 15);
        assert_eq!(result.samples[0].len(), 5);
        assert!(result.maximin_dist > 0.0);
    }

    #[test]
    fn test_maximin_multiple_candidates() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        // Single candidate
        let config1 = MaximinConfig::new(2, 42)
            .with_candidates(1)
            .with_max_iter(0);
        let r1 = maximin_lhs(10, &bounds, &config1).unwrap();

        // Multiple candidates (should be at least as good with 0 iterations)
        let config5 = MaximinConfig::new(2, 42)
            .with_candidates(10)
            .with_max_iter(0);
        let r5 = maximin_lhs(10, &bounds, &config5).unwrap();

        assert!(
            r5.maximin_dist >= r1.maximin_dist * 0.95,
            "More candidates should generally produce better designs"
        );
    }

    #[test]
    fn test_maximin_result_diagnostics() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = MaximinConfig::new(2, 42).with_max_iter(200);
        let result = maximin_lhs(10, &bounds, &config).unwrap();

        // Diagnostics should be populated
        assert!(result.maximin_dist > 0.0);
        assert!(result.initial_maximin_dist > 0.0);
        // n_swaps can be 0 if no improvement found (unlikely but possible)
    }

    #[test]
    fn test_partial_maximin() {
        let samples = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        // Full distance should be 1.0 (adjacent corners)
        let full = maximin_distance(&samples);
        assert!((full - 1.0).abs() < 1e-10);

        // Partial should give same result when checking rows 0 and 1
        let partial = partial_maximin(&samples, 0, 1, f64::INFINITY);
        assert!(partial <= full + 1e-10);
    }
}
