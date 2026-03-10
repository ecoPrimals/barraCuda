// SPDX-License-Identifier: AGPL-3.0-only
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
pub fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1)
}

/// Extract a uniform `f64` in `[0, 1)` from the upper 31 bits of a 64-bit state.
///
/// The upper bits of an LCG have better statistical properties than the lower
/// bits. Extracting 31 bits gives ~2 billion distinct values, which is more
/// than sufficient for Monte Carlo simulations at f64 precision.
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
    reason = "upper-31-bit extraction: max value 2^31-1 fits exactly in f64"
)]
pub fn state_to_f64(state: u64) -> f64 {
    (state >> 33) as f64 / f64::from(u32::MAX)
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
}
