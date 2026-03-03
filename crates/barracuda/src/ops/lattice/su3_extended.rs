// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extended SU(3) WGSL library: reunitarize, exp_cayley, random generation.
//!
//! Depends on complex_f64 + su3 + lcg_f64.

use super::complex_f64::WGSL_COMPLEX64;
use super::lcg::WGSL_LCG_F64;
use super::su3::WGSL_SU3;

/// Raw WGSL source for extended SU(3) operations.
pub const WGSL_SU3_EXTENDED: &str = include_str!("../../shaders/math/su3_extended_f64.wgsl");

/// Build a full preamble: complex_f64 + su3 + lcg + su3_extended.
pub fn su3_extended_preamble() -> String {
    format!("{WGSL_COMPLEX64}\n{WGSL_SU3}\n{WGSL_LCG_F64}\n{WGSL_SU3_EXTENDED}\n")
}
