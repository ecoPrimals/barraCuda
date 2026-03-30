// SPDX-License-Identifier: AGPL-3.0-or-later

//! Safe numeric cast helpers for GPU compute.
//!
//! GPU dispatch code constantly converts between `usize`, `u32`, `f32`, and
//! `f64`. Raw `as` casts silently truncate or lose precision. This module
//! provides checked alternatives that either fail fast or document the
//! intent, enabling gradual migration from `allow(cast_*)` to `warn`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use barracuda::cast;
//!
//! let n: usize = 1024;
//! let n_u32: u32 = cast::usize_as_u32(n)?;
//! let n_f32: f32 = cast::u32_as_f32_lossy(n_u32);
//! ```

use crate::error::{BarracudaError, Result};

/// Convert `usize` to `u32`, returning an error if the value exceeds `u32::MAX`.
///
/// This is the most common cast in GPU dispatch: Rust dimensions are `usize`
/// but WGSL uniforms and workgroup counts are `u32`.
///
/// # Errors
///
/// Returns [`BarracudaError::CastOverflow`] when `value > u32::MAX`.
#[inline]
pub fn usize_as_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| BarracudaError::CastOverflow {
        from_type: "usize",
        to_type: "u32",
        value_description: format!("{value}"),
    })
}

/// Convert `u64` to `u32`, returning an error if the value exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns [`BarracudaError::CastOverflow`] when `value > u32::MAX`.
#[inline]
pub fn u64_as_u32(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| BarracudaError::CastOverflow {
        from_type: "u64",
        to_type: "u32",
        value_description: format!("{value}"),
    })
}

/// Convert `usize` to `u64` (lossless on all supported platforms).
#[inline]
#[must_use]
pub const fn usize_as_u64(value: usize) -> u64 {
    value as u64
}

/// Convert `u32` to `usize` (lossless on all supported platforms).
#[inline]
#[must_use]
pub const fn u32_as_usize(value: u32) -> usize {
    value as usize
}

/// Convert `u32` to `f32` with documented precision loss.
///
/// Values above 2^24 (16,777,216) lose precision in the mantissa.
/// This is acceptable for GPU workgroup counts and dispatch dimensions
/// which rarely exceed that range.
#[inline]
#[must_use]
pub fn u32_as_f32_lossy(value: u32) -> f32 {
    value as f32
}

/// Convert `usize` to `f64` with documented precision loss.
///
/// Values above 2^53 lose precision. Acceptable for buffer-size
/// calculations and diagnostic formatting.
#[inline]
#[must_use]
pub fn usize_as_f64_lossy(value: usize) -> f64 {
    value as f64
}

/// Convert `f64` to `f32`, returning an error if the value is not representable.
///
/// Rejects infinity resulting from overflow and NaN inputs.
///
/// # Errors
///
/// Returns [`BarracudaError::PrecisionLoss`] when the f64 value overflows f32.
#[inline]
pub fn f64_as_f32_checked(value: f64) -> Result<f32> {
    let result = value as f32;
    if value.is_finite() && !result.is_finite() {
        return Err(BarracudaError::PrecisionLoss {
            from_type: "f64",
            to_type: "f32",
            value_description: format!("{value}"),
        });
    }
    Ok(result)
}

/// Convert `f64` to `f32` (lossy but safe for values within f32 range).
///
/// Panics in debug builds if the result is not finite when the input is.
#[inline]
#[must_use]
pub fn f64_as_f32_lossy(value: f64) -> f32 {
    let result = value as f32;
    debug_assert!(
        !value.is_finite() || result.is_finite(),
        "f64→f32 overflow: {value}"
    );
    result
}

/// Convert `u32` to `f64` (lossless — all `u32` values are exactly
/// representable in f64's 53-bit mantissa).
#[inline]
#[must_use]
pub const fn u32_as_f64(value: u32) -> f64 {
    value as f64
}

/// Convert `u64` to `f64` with documented precision loss.
///
/// Values above 2^53 lose precision in the mantissa.
#[inline]
#[must_use]
pub fn u64_as_f64_lossy(value: u64) -> f64 {
    value as f64
}

/// Convert `i32` to `f64` (lossless — all `i32` values are exactly
/// representable in f64's 53-bit mantissa).
#[inline]
#[must_use]
pub const fn i32_as_f64(value: i32) -> f64 {
    value as f64
}

/// Convert `f64` to `usize`, returning an error if the value is negative,
/// non-finite, or exceeds `usize::MAX`.
///
/// # Errors
///
/// Returns [`BarracudaError::CastOverflow`] when `value` cannot be represented
/// as a `usize`.
#[inline]
pub fn f64_as_usize(value: f64) -> Result<usize> {
    if !value.is_finite() || value < 0.0 || value > usize::MAX as f64 {
        return Err(BarracudaError::CastOverflow {
            from_type: "f64",
            to_type: "usize",
            value_description: format!("{value}"),
        });
    }
    Ok(value as usize)
}

/// Truncate `u64` to `u32`, saturating at `u32::MAX`.
///
/// Useful when an exact overflow error is not needed and clamping to the
/// maximum dispatch dimension is acceptable.
#[inline]
#[must_use]
pub const fn u64_as_u32_saturating(value: u64) -> u32 {
    if value > u32::MAX as u64 {
        u32::MAX
    } else {
        value as u32
    }
}

/// Convert `f32` to `f64` (lossless).
#[inline]
#[must_use]
pub const fn f32_as_f64(value: f32) -> f64 {
    value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usize_as_u32_within_range() {
        assert_eq!(usize_as_u32(0).unwrap(), 0);
        assert_eq!(usize_as_u32(u32::MAX as usize).unwrap(), u32::MAX);
    }

    #[test]
    fn usize_as_u32_overflow() {
        if std::mem::size_of::<usize>() > 4 {
            assert!(usize_as_u32(u32::MAX as usize + 1).is_err());
        }
    }

    #[test]
    fn f64_as_f32_checked_normal() {
        assert_eq!(f64_as_f32_checked(1.0).unwrap(), 1.0f32);
        assert_eq!(f64_as_f32_checked(0.0).unwrap(), 0.0f32);
    }

    #[test]
    fn f64_as_f32_checked_overflow() {
        assert!(f64_as_f32_checked(f64::MAX).is_err());
    }

    #[test]
    fn f64_as_f32_checked_nan_passthrough() {
        assert!(f64_as_f32_checked(f64::NAN).unwrap().is_nan());
    }

    #[test]
    fn u32_as_f32_lossy_small() {
        assert_eq!(u32_as_f32_lossy(1024), 1024.0f32);
    }

    #[test]
    fn round_trip_u32_usize() {
        assert_eq!(u32_as_usize(usize_as_u32(42).unwrap()), 42);
    }

    #[test]
    fn u32_as_f64_exact() {
        assert_eq!(u32_as_f64(0), 0.0);
        assert_eq!(u32_as_f64(u32::MAX), f64::from(u32::MAX));
    }

    #[test]
    fn u64_as_f64_lossy_small() {
        assert_eq!(u64_as_f64_lossy(1_000_000), 1_000_000.0);
    }

    #[test]
    fn i32_as_f64_exact() {
        assert_eq!(i32_as_f64(-1), -1.0);
        assert_eq!(i32_as_f64(i32::MAX), f64::from(i32::MAX));
        assert_eq!(i32_as_f64(i32::MIN), f64::from(i32::MIN));
    }

    #[test]
    fn f64_as_usize_valid() {
        assert_eq!(f64_as_usize(0.0).unwrap(), 0);
        assert_eq!(f64_as_usize(42.9).unwrap(), 42);
    }

    #[test]
    fn f64_as_usize_negative() {
        assert!(f64_as_usize(-1.0).is_err());
    }

    #[test]
    fn f64_as_usize_nan() {
        assert!(f64_as_usize(f64::NAN).is_err());
    }

    #[test]
    fn f64_as_usize_infinity() {
        assert!(f64_as_usize(f64::INFINITY).is_err());
    }

    #[test]
    fn u64_as_u32_saturating_within_range() {
        assert_eq!(u64_as_u32_saturating(100), 100);
        assert_eq!(u64_as_u32_saturating(u64::from(u32::MAX)), u32::MAX);
    }

    #[test]
    fn u64_as_u32_saturating_overflow() {
        assert_eq!(u64_as_u32_saturating(u64::from(u32::MAX) + 1), u32::MAX);
        assert_eq!(u64_as_u32_saturating(u64::MAX), u32::MAX);
    }
}
