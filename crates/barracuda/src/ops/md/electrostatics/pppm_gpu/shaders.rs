// SPDX-License-Identifier: AGPL-3.0-only
//! PPPM WGSL shader source strings
//!
//! Extracted from `pppm_gpu` for modularity. All shader sources use `include_str`!
//! for compile-time embedding. Paths are relative to electrostatics directory.

/// B-spline coefficient computation
pub const BSPLINE: &str = include_str!("../bspline.wgsl");

/// Charge spreading from particles to mesh
pub const CHARGE_SPREAD: &str = include_str!("../charge_spread.wgsl");

/// Green's function application in k-space
pub const GREENS_APPLY: &str = include_str!("../greens_apply.wgsl");

/// Force interpolation from mesh to particles
pub const FORCE_INTERP: &str = include_str!("../force_interp.wgsl");

/// Short-range erfc forces
pub const ERFC_FORCES: &str = include_str!("../erfc_forces.wgsl");
