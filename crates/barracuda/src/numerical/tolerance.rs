// SPDX-License-Identifier: AGPL-3.0-or-later
//! Named tolerance tiers for precision guarantees.
//!
//! 13 tiers from exact to qualitative, codifying the precision hierarchy
//! that all spring domains use. Absorbed from groundSpring V76.
//!
//! # Philosophy
//!
//! Every numerical result lives in a precision tier. The tier tells you:
//! - How many significant digits to expect
//! - Whether GPU or CPU produced it
//! - Whether transcendentals were involved
//!
//! # Usage
//!
//! ```
//! use barracuda::numerical::tolerance::Tolerance;
//!
//! let cpu_result = 3.141592653589793_f64;
//! let gpu_result = 3.141592653589780_f64;
//!
//! assert!(Tolerance::GPU_F64.approx_eq(cpu_result, gpu_result));
//! assert!(!Tolerance::Exact.approx_eq(cpu_result, gpu_result));
//! ```
//!
//! # Tier Table
//!
//! | Tier | Abs Tol | Use Case |
//! |------|---------|----------|
//! | `Exact` | 0 | Bitwise identical (integer ops) |
//! | `Ulp1` | ε | FMA rounding |
//! | `Ulp4` | 4ε | Hardware multiply-add chains |
//! | `Digits15` | 1e-15 | CPU f64 arithmetic |
//! | `Digits14` | 1e-14 | DF64 precision ceiling |
//! | `Digits12` | 1e-12 | GPU native f64 accumulation |
//! | `Digits10` | 1e-10 | DF64 with accumulation |
//! | `Digits8` | 1e-8 | GPU f64 + transcendentals |
//! | `Digits6` | 1e-6 | f32 precision |
//! | `Digits4` | 1e-4 | f32 with accumulation |
//! | `Digits3` | 1e-3 | Half precision neighborhood |
//! | `Digits2` | 1e-2 | Coarse / statistical |
//! | `Qualitative` | 0.1 | Sign and order-of-magnitude |

/// Named tolerance tier for precision-aware numerical comparison.
///
/// Tiers are ordered from strictest (`Exact`) to loosest (`Qualitative`).
/// `PartialOrd`/`Ord` reflect this: `Exact < Ulp1 < ... < Qualitative`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tolerance {
    /// Exact: bitwise identical (0 ULP).
    Exact,
    /// Within 1 unit in the last place.
    Ulp1,
    /// Within 4 ULP (hardware FMA rounding chains).
    Ulp4,
    /// 15 digits: 1e-15 (f64 machine epsilon neighborhood).
    Digits15,
    /// 14 digits: 1e-14 (DF64 precision ceiling).
    Digits14,
    /// 12 digits: 1e-12 (GPU native f64 accumulation).
    Digits12,
    /// 10 digits: 1e-10 (DF64 with accumulation).
    Digits10,
    /// 8 digits: 1e-8 (GPU f64 + transcendentals).
    Digits8,
    /// 6 digits: 1e-6 (f32 precision).
    Digits6,
    /// 4 digits: 1e-4 (f32 with accumulation).
    Digits4,
    /// 3 digits: 1e-3 (half precision neighborhood).
    Digits3,
    /// 2 digits: 1e-2 (coarse / statistical).
    Digits2,
    /// Qualitative: 0.1 (sign and order-of-magnitude only).
    Qualitative,
}

impl Tolerance {
    /// Absolute tolerance value for this tier.
    #[must_use]
    pub const fn abs_tol(&self) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::Ulp1 => f64::EPSILON,
            Self::Ulp4 => 4.0 * f64::EPSILON,
            Self::Digits15 => 1e-15,
            Self::Digits14 => 1e-14,
            Self::Digits12 => 1e-12,
            Self::Digits10 => 1e-10,
            Self::Digits8 => 1e-8,
            Self::Digits6 => 1e-6,
            Self::Digits4 => 1e-4,
            Self::Digits3 => 1e-3,
            Self::Digits2 => 1e-2,
            Self::Qualitative => 0.1,
        }
    }

    /// Whether two f64 values are equal within this tolerance tier.
    #[must_use]
    pub fn approx_eq(self, a: f64, b: f64) -> bool {
        if self == Self::Exact {
            return a == b;
        }
        (a - b).abs() <= self.abs_tol()
    }

    /// Whether two f64 values are equal within this tolerance tier,
    /// using relative tolerance for values far from zero.
    ///
    /// `|a - b| <= tol * max(|a|, |b|, 1.0)`
    #[must_use]
    pub fn approx_eq_rel(self, a: f64, b: f64) -> bool {
        if self == Self::Exact {
            return a == b;
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= self.abs_tol() * scale
    }

    /// Recommended tolerance for CPU f64 arithmetic.
    pub const CPU_F64: Self = Self::Digits15;

    /// Recommended tolerance for GPU native f64.
    pub const GPU_F64: Self = Self::Digits12;

    /// Recommended tolerance for DF64 (f32-pair emulation, ~14 digits).
    pub const DF64: Self = Self::Digits10;

    /// Recommended tolerance for GPU f64 with transcendentals.
    pub const GPU_TRANSCENDENTAL: Self = Self::Digits8;

    /// Recommended tolerance for f32 precision.
    pub const F32: Self = Self::Digits6;

    /// Loosen by one tier (returns `Qualitative` if already at loosest).
    #[must_use]
    pub const fn loosen(&self) -> Self {
        match self {
            Self::Exact => Self::Ulp1,
            Self::Ulp1 => Self::Ulp4,
            Self::Ulp4 => Self::Digits15,
            Self::Digits15 => Self::Digits14,
            Self::Digits14 => Self::Digits12,
            Self::Digits12 => Self::Digits10,
            Self::Digits10 => Self::Digits8,
            Self::Digits8 => Self::Digits6,
            Self::Digits6 => Self::Digits4,
            Self::Digits4 => Self::Digits3,
            Self::Digits3 => Self::Digits2,
            Self::Digits2 => Self::Qualitative,
            Self::Qualitative => Self::Qualitative,
        }
    }

    /// Number of tiers (for iteration / testing).
    pub const COUNT: usize = 13;
}

impl std::fmt::Display for Tolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact (0 ULP)"),
            Self::Ulp1 => write!(f, "1 ULP ({:.2e})", f64::EPSILON),
            Self::Ulp4 => write!(f, "4 ULP ({:.2e})", 4.0 * f64::EPSILON),
            _ => write!(f, "{:.0e}", self.abs_tol()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering() {
        assert!(Tolerance::Exact < Tolerance::Ulp1);
        assert!(Tolerance::Ulp1 < Tolerance::Digits15);
        assert!(Tolerance::Digits15 < Tolerance::Qualitative);
    }

    #[test]
    fn exact_requires_bitwise() {
        assert!(Tolerance::Exact.approx_eq(1.0, 1.0));
        assert!(!Tolerance::Exact.approx_eq(1.0, 1.0 + f64::EPSILON));
    }

    #[test]
    fn ulp1_accepts_epsilon() {
        let a = 1.0_f64;
        let b = a + f64::EPSILON;
        assert!(Tolerance::Ulp1.approx_eq(a, b));
    }

    #[test]
    fn gpu_f64_tier() {
        let cpu = std::f64::consts::PI;
        #[expect(
            clippy::approx_constant,
            reason = "intentionally truncated pi to simulate GPU f64 precision loss"
        )]
        let gpu = 3.141_592_653_590_f64;
        assert!(Tolerance::GPU_F64.approx_eq(cpu, gpu));
    }

    #[test]
    fn f32_tier_accepts_6_digits() {
        let a = 1.234_567_f64;
        let b = 1.234_567_5_f64;
        assert!(Tolerance::F32.approx_eq(a, b));
    }

    #[test]
    fn loosen_chain_reaches_qualitative() {
        let mut t = Tolerance::Exact;
        for _ in 0..20 {
            t = t.loosen();
        }
        assert_eq!(t, Tolerance::Qualitative);
    }

    #[test]
    fn relative_tolerance() {
        let a = 1e10_f64;
        let b = a + 1.0;
        // Absolute: |a - b| = 1.0, tol = 1e-12 → fails
        assert!(!Tolerance::Digits12.approx_eq(a, b));
        // Relative: |a - b| / max(|a|, |b|, 1) = 1e-10, tol = 1e-10 → passes
        assert!(Tolerance::Digits10.approx_eq_rel(a, b));
    }

    #[test]
    fn display_exact() {
        let exact = Tolerance::Exact;
        assert_eq!(format!("{exact}"), "exact (0 ULP)");
    }

    #[test]
    fn count_is_correct() {
        assert_eq!(Tolerance::COUNT, 13);
    }
}
