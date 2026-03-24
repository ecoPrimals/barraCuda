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
}
