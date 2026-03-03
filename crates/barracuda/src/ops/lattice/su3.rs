// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL SU(3) matrix algebra library.
//!
//! Provides `WGSL_SU3` and `su3_preamble()` for building shaders that require
//! SU(3) gauge-field algebra.  Always prepend `WGSL_COMPLEX64` first.
//!
//! For hybrid DF64 core-streaming shaders, use `su3_df64_preamble()` which
//! includes both the f64 and DF64 SU(3) libraries. The hybrid shaders use
//! DF64 arithmetic on FP32 cores for bulk matmuls and fall back to native
//! f64 for precision-critical operations (algebra projection, reductions).

use super::complex_f64::WGSL_COMPLEX64;

/// Raw WGSL source for SU(3) 3×3 complex matrix algebra.
///
/// Depends on `complex_f64.wgsl` definitions — always use `su3_preamble()`
/// or prepend `WGSL_COMPLEX64` manually.
pub const WGSL_SU3: &str = include_str!("../../shaders/math/su3.wgsl");

/// Raw WGSL source for the DF64 (f32-pair) arithmetic library.
pub const WGSL_DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");

/// Raw WGSL source for DF64 transcendental functions (exp, log, sqrt, sin, cos, etc.)
pub const WGSL_DF64_TRANSCENDENTALS: &str =
    include_str!("../../shaders/math/df64_transcendentals.wgsl");

/// Raw WGSL source for DF64 SU(3) matrix algebra (complex + matrix ops).
///
/// Depends on `df64_core.wgsl` — always use `su3_df64_preamble()`.
pub const WGSL_SU3_DF64: &str = include_str!("../../shaders/math/su3_df64.wgsl");

/// Build a complete shader preamble: complex_f64 + su3.
///
/// Call once per shader; append the domain shader source after this string.
pub fn su3_preamble() -> String {
    format!("{WGSL_COMPLEX64}\n{WGSL_SU3}\n")
}

/// Build a hybrid DF64 shader preamble: complex_f64 + su3 + df64_core + su3_df64.
///
/// Includes both native f64 SU(3) ops (for algebra projection) and DF64 SU(3)
/// ops (for bulk matmuls on FP32 cores). Use for hybrid precision shaders.
pub fn su3_df64_preamble() -> String {
    format!("{WGSL_COMPLEX64}\n{WGSL_SU3}\n{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{WGSL_SU3_DF64}\n")
}
