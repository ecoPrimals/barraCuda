// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL LCG PRNG library for lattice GPU kernels.
//!
//! Provides `WGSL_LCG_F64` and `lcg_preamble()` for building shaders that
//! require per-thread pseudorandom number generation.

/// Raw WGSL source for LCG f64 PRNG (`lcg_step`, `lcg_uniform`, `lcg_gaussian`).
pub const WGSL_LCG_F64: &str = include_str!("../../shaders/math/lcg_f64.wgsl");

/// Build preamble: `lcg_f64` (standalone, no dependencies).
#[must_use]
pub fn lcg_preamble() -> String {
    WGSL_LCG_F64.to_string()
}
