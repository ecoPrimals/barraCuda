// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL f64 complex arithmetic library.
//!
//! Provides the `WGSL_COMPLEX64` string constant that must be prepended to any
//! shader requiring complex arithmetic.  Use `complex_preamble()` to obtain a
//! complete shader source string.

/// Raw WGSL source for f64 complex arithmetic.
///
/// Prepend to any WGSL shader that needs `c64_*` functions.
/// All operations on `vec2<f64>` where `.x = Re`, `.y = Im`.
pub const WGSL_COMPLEX64: &str = include_str!("../../shaders/math/complex_f64.wgsl");

/// Prepend the complex-f64 preamble to a shader source string.
///
/// ```rust
/// use barracuda::ops::lattice::complex_f64::prepend_complex;
/// let full_src = prepend_complex("// my shader\nfn foo() {}");
/// assert!(full_src.starts_with("// complex_f64"));
/// ```
#[must_use]
pub fn prepend_complex(shader: &str) -> String {
    format!("{WGSL_COMPLEX64}\n{shader}")
}
