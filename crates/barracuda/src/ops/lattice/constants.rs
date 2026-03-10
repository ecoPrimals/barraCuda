// SPDX-License-Identifier: AGPL-3.0-only

//! Centralized constants for lattice field theory modules.
//!
//! Collects LCG PRNG parameters, SU(3) color constants, and numerical
//! guards used across `su3`, `wilson`, `dirac`, `cg`, and `pseudofermion`.
//!
//! Absorbed from hotSpring v0.64 `lattice/constants.rs` (Feb 2026).

/// LCG multiplier (Knuth MMIX).
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// LCG increment (Knuth MMIX).
pub const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Mantissa bits for LCG → uniform [0, 1) conversion: 53-bit precision.
pub const LCG_53_DIVISOR: f64 = (1u64 << 53) as f64;

/// Division guard for lattice CG/reunitarization.
pub const LATTICE_DIVISION_GUARD: f64 = 1e-30;

/// Default maximum CG solver iterations for D†D inversions.
pub const CG_MAX_ITER_DEFAULT: usize = 5000;

/// Default CG convergence tolerance for lattice solvers.
pub const CG_TOL_DEFAULT: f64 = 1e-8;

/// Advance the LCG state by one step.
#[inline]
pub const fn lcg_step(seed: &mut u64) {
    *seed = seed
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
}

/// Generate a uniform f64 in [0, 1) from 53 bits of LCG state.
#[inline]
pub fn lcg_uniform_f64(seed: &mut u64) -> f64 {
    lcg_step(seed);
    (*seed >> 11) as f64 / LCG_53_DIVISOR
}

/// Box-Muller Gaussian deviate N(0, 1) from two LCG draws.
#[inline]
pub fn lcg_gaussian(seed: &mut u64) -> f64 {
    let u1 = lcg_uniform_f64(seed);
    let u2 = lcg_uniform_f64(seed);
    (-2.0 * u1.max(LATTICE_DIVISION_GUARD).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_step_deterministic() {
        let mut a = 42u64;
        let mut b = 42u64;
        lcg_step(&mut a);
        lcg_step(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn lcg_uniform_in_range() {
        let mut seed = 12345u64;
        for _ in 0..1000 {
            let v = lcg_uniform_f64(&mut seed);
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn lcg_gaussian_is_finite() {
        let mut seed = 99u64;
        for _ in 0..1000 {
            let g = lcg_gaussian(&mut seed);
            assert!(g.is_finite(), "Gaussian deviate must be finite: {g}");
        }
    }
}
