// SPDX-License-Identifier: AGPL-3.0-only
//! Precision guard constants for GPU numerical safety.
//!
//! GPU f64 division-by-zero is the primary NaN source across all spring
//! domains (groundSpring V76 discovery). These constants provide safe
//! floors for denominators, logarithm arguments, and square root inputs.
//!
//! # Usage in WGSL
//!
//! Inject the preamble via [`WGSL_PREAMBLE`] or [`WGSL_PREAMBLE_F32`]:
//!
//! ```wgsl
//! let safe = a / max(b, EPS_SAFE_DIV);
//! let log_safe = log(max(x, EPS_SAFE_LOG));
//! ```
//!
//! # Rust-side validation
//!
//! Use the Rust constants for CPU reference implementations and tolerance
//! assertions that mirror GPU behavior.

/// Safe divisor floor — prevents division-by-zero NaN.
/// `a / max(b, SAFE_DIV)` guarantees a finite result.
pub const SAFE_DIV: f64 = 1e-300;

/// Safe log argument floor — `log(max(x, SAFE_LOG))` prevents -Inf.
pub const SAFE_LOG: f64 = 1e-300;

/// Safe sqrt argument floor — `sqrt(max(x, SAFE_SQRT))` prevents NaN
/// from negative inputs due to floating-point drift.
pub const SAFE_SQRT: f64 = 0.0;

/// Underflow guard for exponentials — `exp(max(x, UNDERFLOW_GUARD))`.
/// Prevents denormalized results that trigger slow paths on some GPUs.
pub const UNDERFLOW_GUARD: f64 = -700.0;

/// Sum-of-squares accumulator floor for numerical stability.
/// Prevents catastrophic cancellation when accumulating squared
/// deviations near zero.
pub const SSA_FLOOR: f64 = 1e-30;

/// f32 variant: safe divisor floor.
pub const SAFE_DIV_F32: f32 = 1e-38;

/// f32 variant: safe log argument floor.
pub const SAFE_LOG_F32: f32 = 1e-38;

/// f32 variant: underflow guard for exponentials.
pub const UNDERFLOW_GUARD_F32: f32 = -87.0;

/// f32 variant: SSA floor.
pub const SSA_FLOOR_F32: f32 = 1e-15;

/// WGSL preamble injecting f64 precision guard constants into shaders.
///
/// Shaders that perform division, log, sqrt, or exp on potentially
/// degenerate inputs should use these instead of raw operations:
///
/// ```wgsl
/// let safe = a / max(b, EPS_SAFE_DIV);
/// ```
pub const WGSL_PREAMBLE: &str = "\
// eps:: precision guards (f64) — groundSpring V76 absorption
const EPS_SAFE_DIV: f64 = f64(1e-300);
const EPS_SAFE_LOG: f64 = f64(1e-300);
const EPS_SAFE_SQRT: f64 = f64(0.0);
const EPS_UNDERFLOW_GUARD: f64 = f64(-700.0);
const EPS_SSA_FLOOR: f64 = f64(1e-30);
";

/// WGSL preamble for f32/DF64 shaders.
pub const WGSL_PREAMBLE_F32: &str = "\
// eps:: precision guards (f32) — groundSpring V76 absorption
const EPS_SAFE_DIV: f32 = 1e-38;
const EPS_SAFE_LOG: f32 = 1e-38;
const EPS_SAFE_SQRT: f32 = 0.0;
const EPS_UNDERFLOW_GUARD: f32 = -87.0;
const EPS_SSA_FLOOR: f32 = 1e-15;
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_div_prevents_nan() {
        let a = 1.0_f64;
        let b = 0.0_f64;
        let result = a / b.max(SAFE_DIV);
        assert!(result.is_finite());
    }

    #[test]
    fn safe_log_prevents_neg_inf() {
        let x = 0.0_f64;
        let result = x.max(SAFE_LOG).ln();
        assert!(result.is_finite());
    }

    #[test]
    fn underflow_guard_prevents_denorm() {
        let x = -800.0_f64;
        let result = x.max(UNDERFLOW_GUARD).exp();
        assert!(result >= 0.0);
    }

    #[test]
    fn wgsl_preamble_contains_all_constants() {
        assert!(WGSL_PREAMBLE.contains("EPS_SAFE_DIV"));
        assert!(WGSL_PREAMBLE.contains("EPS_SAFE_LOG"));
        assert!(WGSL_PREAMBLE.contains("EPS_SAFE_SQRT"));
        assert!(WGSL_PREAMBLE.contains("EPS_UNDERFLOW_GUARD"));
        assert!(WGSL_PREAMBLE.contains("EPS_SSA_FLOOR"));
    }

    #[test]
    fn f32_preamble_uses_f32_values() {
        assert!(WGSL_PREAMBLE_F32.contains("f32"));
        assert!(!WGSL_PREAMBLE_F32.contains("f64"));
    }
}
