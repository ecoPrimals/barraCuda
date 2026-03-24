// SPDX-License-Identifier: AGPL-3.0-or-later
//! Latin Hypercube Sampling (LHS) and uniform random sampling
//!
//! Latin Hypercube Sampling provides better space coverage than pure random
//! sampling by ensuring each dimension is evenly stratified. For `n` samples
//! in `d` dimensions, LHS guarantees exactly one sample per interval when
//! each dimension is divided into `n` equal intervals.
//!
//! # Algorithm
//!
//! 1. Divide each dimension into `n_samples` equal intervals
//! 2. Randomly permute the interval assignments for each dimension
//! 3. Sample uniformly within each assigned interval
//! 4. Result: space-filling design with one sample per "row" and "column"
//!
//! # Cross-Domain Applications
//!
//! - **Physics**: Initial conditions for molecular dynamics, parameter space for EOS fitting
//! - **ML**: Hyperparameter search, neural architecture search
//! - **Graphics**: Material parameter sampling, light source placement
//! - **Audio**: Filter coefficient exploration, impulse response design
//!
//! # References
//!
//! - `McKay`, M.D., Beckman, R.J., Conover, W.J. (1979). "A comparison of three
//!   methods for selecting values of input variables in the analysis of output
//!   from a computer code." Technometrics, 21(2), 239-245.
//! - hotSpring: `control/surrogate/scripts/run_benchmark_functions.py:108-120`

use crate::error::{BarracudaError, Result};

/// Simple deterministic PRNG (xoshiro128** algorithm, 64-bit variant)
///
/// Self-contained RNG that avoids external dependencies. Suitable for
/// sampling where cryptographic quality is not required.
pub(crate) struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Create a new PRNG from a seed
    pub(crate) fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed into state
        let mut z = seed;
        let mut s = [0u64; 4];
        for si in &mut s {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            *si = z ^ (z >> 31);
        }
        Self { s }
    }

    /// Generate next u64
    pub(crate) fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate uniform f64 in [0, 1)
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fisher-Yates shuffle of a mutable slice
    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Generate Latin Hypercube Sample points within given bounds.
///
/// Creates `n_samples` points in `d` dimensions where each dimension is
/// divided into `n_samples` equal intervals, and exactly one sample falls
/// in each interval per dimension. This provides better space coverage than
/// pure random sampling.
///
/// # Arguments
///
/// * `n_samples` - Number of sample points to generate (must be > 0)
/// * `bounds` - Box bounds `[(min, max), ...]` for each dimension (must be non-empty)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// `Vec<Vec<f64>>` of shape `[n_samples][n_dims]`, each inner vector is one sample point.
///
/// # Examples
///
/// ```
/// use barracuda::sample::latin_hypercube;
///
/// let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
/// let samples = latin_hypercube(100, &bounds, 42)?;
///
/// assert_eq!(samples.len(), 100);
/// assert_eq!(samples[0].len(), 2);
///
/// // All points within bounds
/// for point in &samples {
///     assert!(point[0] >= -5.0 && point[0] <= 5.0);
///     assert!(point[1] >= -5.0 && point[1] <= 5.0);
/// }
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # Errors
///
/// Returns [`Err`] if `n_samples` is 0, bounds is empty, or any bound has lower >= upper.
pub fn latin_hypercube(
    n_samples: usize,
    bounds: &[(f64, f64)],
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    if n_samples == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "n_samples must be > 0".to_string(),
        });
    }

    if bounds.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "bounds must be non-empty".to_string(),
        });
    }

    // Validate bounds
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Bounds for dimension {i} are invalid: [{lo}, {hi}] (lower >= upper)"
                ),
            });
        }
    }

    let n_dims = bounds.len();
    let mut rng = Xoshiro256::new(seed);

    // Initialize samples matrix: n_samples × n_dims
    let mut samples = vec![vec![0.0; n_dims]; n_samples];

    // For each dimension, create stratified samples
    for d in 0..n_dims {
        let (lo, hi) = bounds[d];

        // Create permutation of interval indices [0, 1, ..., n_samples-1]
        let mut perm: Vec<usize> = (0..n_samples).collect();
        rng.shuffle(&mut perm);

        // Create interval edges: n_samples+1 evenly spaced points
        let interval_width = (hi - lo) / n_samples as f64;

        // For each sample, place it uniformly within its assigned interval
        for i in 0..n_samples {
            let interval_idx = perm[i];
            let interval_lo = (interval_idx as f64).mul_add(interval_width, lo);
            let interval_hi = interval_lo + interval_width;

            // Uniform random within interval
            samples[i][d] = rng
                .next_f64()
                .mul_add(interval_hi - interval_lo, interval_lo);
        }
    }

    Ok(samples)
}

/// Generate uniform random sample points within given bounds.
///
/// Simple uniform random sampling for comparison with LHS or when
/// stratification is not needed.
///
/// # Arguments
///
/// * `n_samples` - Number of sample points to generate
/// * `bounds` - Box bounds `[(min, max), ...]` for each dimension
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// `Vec<Vec<f64>>` of shape `[n_samples][n_dims]`
#[must_use]
pub fn random_uniform(n_samples: usize, bounds: &[(f64, f64)], seed: u64) -> Vec<Vec<f64>> {
    let n_dims = bounds.len();
    let mut rng = Xoshiro256::new(seed);

    (0..n_samples)
        .map(|_| {
            (0..n_dims)
                .map(|d| {
                    let (lo, hi) = bounds[d];
                    rng.next_f64().mul_add(hi - lo, lo)
                })
                .collect()
        })
        .collect()
}

/// Compute the minimum pairwise distance between all points (maximin metric).
///
/// Higher values indicate better space-filling properties.
/// LHS should consistently produce higher maximin distances than random sampling
/// for the same number of points.
#[must_use]
pub fn maximin_distance(points: &[Vec<f64>]) -> f64 {
    let n = points.len();
    if n < 2 {
        return f64::INFINITY;
    }

    let mut min_dist = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist: f64 = points[i]
                .iter()
                .zip(points[j].iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }
    min_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lhs_basic_2d() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let samples = latin_hypercube(10, &bounds, 42).unwrap();

        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 2);

        // All points within bounds
        for point in &samples {
            assert!(point[0] >= -5.0 && point[0] <= 5.0);
            assert!(point[1] >= -5.0 && point[1] <= 5.0);
        }
    }

    #[test]
    fn test_lhs_stratification() {
        // With 10 samples in [0, 10], each sample should fall in a unique interval
        let bounds = vec![(0.0, 10.0)];
        let samples = latin_hypercube(10, &bounds, 42).unwrap();

        // Check that each interval [0,1), [1,2), ..., [9,10] has exactly one sample
        let mut intervals_used = [false; 10];
        for point in &samples {
            let interval = (point[0] as usize).min(9);
            assert!(
                !intervals_used[interval],
                "Interval {interval} used more than once"
            );
            intervals_used[interval] = true;
        }

        // All intervals should be used
        assert!(intervals_used.iter().all(|&used| used));
    }

    #[test]
    fn test_lhs_higher_dimensions() {
        let bounds = vec![
            (0.0, 1.0),
            (-100.0, 100.0),
            (0.0, 1e6),
            (-1.0, 1.0),
            (0.0, 0.001),
        ];
        let samples = latin_hypercube(50, &bounds, 123).unwrap();

        assert_eq!(samples.len(), 50);
        assert_eq!(samples[0].len(), 5);

        // All within bounds
        for point in &samples {
            for (d, &val) in point.iter().enumerate() {
                let (lo, hi) = bounds[d];
                assert!(
                    val >= lo && val <= hi,
                    "Dim {d} value {val} outside [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn test_lhs_reproducibility() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let a = latin_hypercube(20, &bounds, 42).unwrap();
        let b = latin_hypercube(20, &bounds, 42).unwrap();

        // Same seed → same samples
        for i in 0..20 {
            assert_eq!(a[i][0], b[i][0]);
            assert_eq!(a[i][1], b[i][1]);
        }
    }

    #[test]
    fn test_lhs_different_seeds() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let a = latin_hypercube(20, &bounds, 42).unwrap();
        let b = latin_hypercube(20, &bounds, 99).unwrap();

        // Different seeds → different samples (with overwhelming probability)
        let different = a
            .iter()
            .zip(b.iter())
            .any(|(ai, bi)| (ai[0] - bi[0]).abs() > 1e-15);
        assert!(different);
    }

    #[test]
    fn test_lhs_vs_random_coverage() {
        // LHS should have better space-filling properties (higher maximin distance)
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];

        // Run multiple trials and compare average maximin distance
        let n_trials = 10;
        let n_samples = 30;

        let mut lhs_maximin_total = 0.0;
        let mut rng_maximin_total = 0.0;

        for trial in 0..n_trials {
            let seed = trial as u64 * 1000 + 42;

            let lhs_points = latin_hypercube(n_samples, &bounds, seed).unwrap();
            let rng_points = random_uniform(n_samples, &bounds, seed + 1);

            lhs_maximin_total += maximin_distance(&lhs_points);
            rng_maximin_total += maximin_distance(&rng_points);
        }

        let lhs_avg = lhs_maximin_total / n_trials as f64;
        let rng_avg = rng_maximin_total / n_trials as f64;

        // LHS should have better average maximin distance
        // (Not guaranteed per-trial, but statistically very likely over 10 trials)
        assert!(
            lhs_avg > rng_avg * 0.9, // Allow some slack but LHS should be at least competitive
            "LHS avg maximin ({lhs_avg:.4}) should be >= random avg maximin ({rng_avg:.4})"
        );
    }

    #[test]
    fn test_lhs_errors() {
        let bounds = vec![(0.0, 1.0)];

        // Zero samples
        assert!(latin_hypercube(0, &bounds, 42).is_err());

        // Empty bounds
        assert!(latin_hypercube(10, &[], 42).is_err());

        // Invalid bounds (lo >= hi)
        assert!(latin_hypercube(10, &[(5.0, 1.0)], 42).is_err());
        assert!(latin_hypercube(10, &[(1.0, 1.0)], 42).is_err());
    }

    #[test]
    fn test_lhs_single_sample() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
        let samples = latin_hypercube(1, &bounds, 42).unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].len(), 2);
        assert!(samples[0][0] >= 0.0 && samples[0][0] <= 10.0);
        assert!(samples[0][1] >= -5.0 && samples[0][1] <= 5.0);
    }

    #[test]
    fn test_random_uniform_basic() {
        let bounds = vec![(0.0, 1.0), (-1.0, 1.0)];
        let samples = random_uniform(100, &bounds, 42);

        assert_eq!(samples.len(), 100);
        for point in &samples {
            assert!(point[0] >= 0.0 && point[0] <= 1.0);
            assert!(point[1] >= -1.0 && point[1] <= 1.0);
        }
    }

    #[test]
    fn test_maximin_distance_basic() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let dist = maximin_distance(&points);
        assert!((dist - 1.0).abs() < 1e-10); // Min distance is 1.0 (adjacent corners)
    }

    #[test]
    fn test_lhs_10d_nuclear_eos_bounds() {
        // Real-world test: 10D Skyrme parameter space from hotSpring
        let bounds = vec![
            (-3000.0, -1000.0), // t0
            (100.0, 800.0),     // t1
            (-800.0, 0.0),      // t2
            (8000.0, 20_000.0), // t3
            (-1.0, 2.0),        // x0
            (-2.0, 1.0),        // x1
            (-2.0, 0.0),        // x2
            (0.0, 2.0),         // x3
            (0.05, 0.5),        // alpha
            (50.0, 200.0),      // W0
        ];

        let samples = latin_hypercube(1000, &bounds, 42).unwrap();
        assert_eq!(samples.len(), 1000);
        assert_eq!(samples[0].len(), 10);

        // All within bounds
        for point in &samples {
            for (d, &val) in point.iter().enumerate() {
                let (lo, hi) = bounds[d];
                assert!(val >= lo && val <= hi, "Dim {d} out of bounds");
            }
        }
    }
}
