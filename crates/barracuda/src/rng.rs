// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic CPU PRNG for reproducible simulations.
//!
//! Provides a Knuth LCG (Linear Congruential Generator) with 64-bit state.
//! All modules needing deterministic pseudo-random sequences should use these
//! functions rather than duplicating the multiplier constant.
//!
//! For GPU PRNG, see `ops::prng_xoshiro_wgsl` (xoshiro128** via WGSL).
//! The GPU and CPU generators are intentionally different — GPU uses u32-only
//! xoshiro128** for portability (WGSL lacks u64), while CPU uses u64 LCG for
//! longer period and better statistical properties on scalar workloads.
//!
//! Absorbed from healthSpring `rng.rs` (V13) — centralizes the constant that
//! was duplicated across 4+ spring modules.

/// Knuth LCG multiplier (64-bit).
///
/// This is the multiplier from Knuth's TAOCP Vol 2, used with an additive
/// constant of 1. Period is 2^64.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// Advance the LCG state by one step.
///
/// `state_{n+1} = state_n × LCG_MULTIPLIER + 1 (mod 2^64)`
#[must_use]
#[inline]
pub const fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1)
}

/// Extract a uniform `f64` in `[0, 1)` from the upper 53 bits of a 64-bit state.
///
/// The upper bits of an LCG have better statistical properties than the lower
/// bits. Extracting 53 bits matches f64 mantissa precision (52+1 implicit),
/// giving ~9 quadrillion distinct values.
///
/// # Example
///
/// ```
/// use barracuda::rng::{lcg_step, state_to_f64};
///
/// let state = lcg_step(42);
/// let value = state_to_f64(state);
/// assert!((0.0..1.0).contains(&value));
/// ```
#[must_use]
#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "upper-53-bit extraction: fits exactly in f64 mantissa"
)]
pub fn state_to_f64(state: u64) -> f64 {
    (state >> 11) as f64 / (1u64 << 53) as f64
}

/// Generate `n` uniform f64 values in `[0, 1)` from a seed.
///
/// Deterministic: same seed always produces the same sequence.
#[must_use]
pub fn uniform_f64_sequence(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = lcg_step(state);
            state_to_f64(state)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// 32-bit LCG (ludoSpring contract, Mar 2026)
// ═══════════════════════════════════════════════════════════════════

/// 32-bit LCG multiplier (Knuth MMIX variant).
///
/// `state_{n+1} = state_n × 1664525 + 1013904223 (mod 2^32)`
///
/// Period is 2^32. Used by ludoSpring for game-speed procedural generation
/// where full u64 state is unnecessary.
pub const LCG32_MULTIPLIER: u32 = 1_664_525;

/// 32-bit LCG additive constant (Numerical Recipes).
pub const LCG32_INCREMENT: u32 = 1_013_904_223;

/// Advance the 32-bit LCG state by one step.
///
/// `state_{n+1} = state_n × 1664525 + 1013904223 (mod 2^32)`
#[must_use]
#[inline]
pub const fn lcg_step_u32(state: u32) -> u32 {
    state
        .wrapping_mul(LCG32_MULTIPLIER)
        .wrapping_add(LCG32_INCREMENT)
}

/// Extract a uniform `f32` in `[0, 1)` from a 32-bit LCG state.
///
/// Uses the upper 24 bits for best statistical properties (lower bits of
/// LCG have short sub-periods). 24 bits gives 16M distinct values, which
/// exceeds f32's ~7 decimal digit precision.
#[must_use]
#[inline]
pub fn state_to_f32(state: u32) -> f32 {
    (state >> 8) as f32 / 16_777_216.0
}

/// Generate `n` uniform f32 values in `[0, 1)` from a 32-bit seed.
///
/// Deterministic: same seed always produces the same sequence.
#[must_use]
pub fn uniform_f32_sequence(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = lcg_step_u32(state);
            state_to_f32(state)
        })
        .collect()
}

/// Lightweight LCG RNG for reproducible disorder generation.
///
/// Uses the Knuth MMIX full-period LCG (period 2^64). Suitable for
/// Monte Carlo sampling and disordered system construction where
/// cryptographic strength is unnecessary.
///
/// Consolidates the LCG PRNG previously duplicated in `spectral::anderson`
/// and `spectral::lanczos`.
pub struct LcgRng(u64);

impl LcgRng {
    /// Create a new LCG with the given seed (shifted by +1 to avoid zero state).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    const fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(LCG_MULTIPLIER)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Generate a uniform f64 in \[0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_deterministic() {
        let a = lcg_step(42);
        let b = lcg_step(42);
        assert_eq!(a, b);
    }

    #[test]
    fn lcg_different_seeds() {
        assert_ne!(lcg_step(1), lcg_step(2));
    }

    #[test]
    fn state_to_f64_range() {
        let mut state = 12_345_u64;
        for _ in 0..1000 {
            state = lcg_step(state);
            let v = state_to_f64(state);
            assert!((0.0..1.0).contains(&v), "value {v} outside [0, 1)");
        }
    }

    #[test]
    fn uniform_sequence_length() {
        let seq = uniform_f64_sequence(42, 100);
        assert_eq!(seq.len(), 100);
    }

    #[test]
    fn uniform_sequence_deterministic() {
        let a = uniform_f64_sequence(42, 50);
        let b = uniform_f64_sequence(42, 50);
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_sequence_all_in_range() {
        for &v in &uniform_f64_sequence(99, 10_000) {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg32_deterministic() {
        let a = lcg_step_u32(42);
        let b = lcg_step_u32(42);
        assert_eq!(a, b);
    }

    #[test]
    fn lcg32_different_seeds() {
        assert_ne!(lcg_step_u32(1), lcg_step_u32(2));
    }

    #[test]
    fn state_to_f32_range() {
        let mut state = 12_345_u32;
        for _ in 0..1000 {
            state = lcg_step_u32(state);
            let v = state_to_f32(state);
            assert!((0.0..1.0).contains(&v), "f32 value {v} outside [0, 1)");
        }
    }

    #[test]
    fn uniform_f32_sequence_length() {
        let seq = uniform_f32_sequence(42, 100);
        assert_eq!(seq.len(), 100);
    }

    #[test]
    fn uniform_f32_sequence_deterministic() {
        let a = uniform_f32_sequence(42, 50);
        let b = uniform_f32_sequence(42, 50);
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_f32_sequence_all_in_range() {
        for &v in &uniform_f32_sequence(99, 10_000) {
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg32_known_value() {
        let state = lcg_step_u32(0);
        assert_eq!(state, LCG32_INCREMENT);
    }

    #[test]
    fn lcg_rng_uniform_range() {
        let mut rng = LcgRng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn lcg_rng_deterministic() {
        let mut a = LcgRng::new(123);
        let mut b = LcgRng::new(123);
        for _ in 0..100 {
            assert_eq!(a.uniform().to_bits(), b.uniform().to_bits());
        }
    }

    // ── Statistical validation (PRNG YELLOW → GREEN) ────────────

    #[test]
    fn uniform_f64_mean_variance() {
        let n = 100_000;
        let samples = uniform_f64_sequence(42, n);
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!((mean - 0.5).abs() < 0.005, "U(0,1) mean {mean} outside tolerance");
        let expected_var = 1.0 / 12.0;
        assert!(
            (var - expected_var).abs() < 0.005,
            "U(0,1) variance {var} vs expected {expected_var}"
        );
    }

    #[test]
    fn uniform_f32_mean_variance() {
        let n = 100_000;
        let samples = uniform_f32_sequence(42, n);
        let mean = samples.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
        let var = samples
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n as f64;

        assert!((mean - 0.5).abs() < 0.005, "U(0,1) f32 mean {mean} outside tolerance");
        let expected_var = 1.0 / 12.0;
        assert!(
            (var - expected_var).abs() < 0.005,
            "U(0,1) f32 variance {var} vs expected {expected_var}"
        );
    }

    #[test]
    fn uniform_f64_chi_squared_bins() {
        let n = 100_000;
        let n_bins = 10;
        let samples = uniform_f64_sequence(7, n);
        let mut bins = vec![0u64; n_bins];
        for &x in &samples {
            let bin = (x * n_bins as f64).min((n_bins - 1) as f64) as usize;
            bins[bin] += 1;
        }

        let expected = n as f64 / n_bins as f64;
        let chi2: f64 = bins.iter().map(|&b| (b as f64 - expected).powi(2) / expected).sum();
        // df = n_bins - 1 = 9; chi2 critical at p=0.001 is ~27.9
        assert!(
            chi2 < 30.0,
            "chi-squared {chi2} exceeds critical value — uniform distribution suspect"
        );
    }

    #[test]
    fn lcg_rng_mean_variance() {
        let n = 100_000;
        let mut rng = LcgRng::new(42);
        let samples: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!((mean - 0.5).abs() < 0.005, "LcgRng mean {mean} outside tolerance");
        let expected_var = 1.0 / 12.0;
        assert!(
            (var - expected_var).abs() < 0.005,
            "LcgRng variance {var} vs expected {expected_var}"
        );
    }

    #[test]
    fn cpu_box_muller_gaussian_moments() {
        let n = 100_000;
        let mut seed = 42u64;
        let samples: Vec<f64> = (0..n)
            .map(|_| super::super::ops::lattice::constants::lcg_gaussian(&mut seed))
            .collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let skew = samples.iter().map(|x| ((x - mean) / var.sqrt()).powi(3)).sum::<f64>()
            / n as f64;
        let kurt = samples.iter().map(|x| ((x - mean) / var.sqrt()).powi(4)).sum::<f64>()
            / n as f64
            - 3.0;

        assert!(mean.abs() < 0.02, "N(0,1) mean {mean} outside tolerance");
        assert!((var - 1.0).abs() < 0.05, "N(0,1) variance {var} outside tolerance");
        assert!(skew.abs() < 0.1, "N(0,1) skewness {skew} outside tolerance");
        assert!(kurt.abs() < 0.2, "N(0,1) excess kurtosis {kurt} outside tolerance");
    }

    #[test]
    fn cpu_box_muller_chi_squared() {
        let n = 100_000;
        let mut seed = 99u64;
        let samples: Vec<f64> = (0..n)
            .map(|_| super::super::ops::lattice::constants::lcg_gaussian(&mut seed))
            .collect();

        let n_bins = 20;
        let bin_width = 0.5_f64;
        let range_min = -5.0_f64;
        let mut bins = vec![0u64; n_bins];
        for &x in &samples {
            let idx = ((x - range_min) / bin_width) as i64;
            if (0..n_bins as i64).contains(&idx) {
                bins[idx as usize] += 1;
            }
        }

        let sigma = 1.0;
        let expected_fracs: Vec<f64> = (0..n_bins)
            .map(|i| {
                let lo = range_min + i as f64 * bin_width;
                let hi = lo + bin_width;
                0.5 * (erf_approx(hi / (sigma * std::f64::consts::SQRT_2))
                    - erf_approx(lo / (sigma * std::f64::consts::SQRT_2)))
            })
            .collect();

        let mut chi2 = 0.0;
        for (i, &observed) in bins.iter().enumerate() {
            let expected = expected_fracs[i] * n as f64;
            if expected > 5.0 {
                chi2 += (observed as f64 - expected).powi(2) / expected;
            }
        }
        // df ≈ 15 usable bins; chi2 critical at p=0.001 ≈ 37
        assert!(
            chi2 < 40.0,
            "Gaussian chi-squared {chi2} exceeds critical value — distribution suspect"
        );
    }

    #[test]
    fn multiple_seeds_independent() {
        let seqs: Vec<Vec<f64>> = (0..10).map(|s| uniform_f64_sequence(s * 1000 + 1, 1000)).collect();
        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "seeds {i} and {j} produced identical sequences");
            }
        }
    }

    fn erf_approx(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let result = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 { result } else { -result }
    }
}
