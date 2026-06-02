// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kinetic plasma / dielectric GPU shaders (Paper 44–45).

/// BGK relaxation for multi-species kinetic plasma.
pub const WGSL_BGK_RELAXATION: &str =
    include_str!("../../shaders/science/plasma/bgk_relaxation_f64.wgsl");

/// 1D Euler fluid update with HLL Riemann solver.
pub const WGSL_EULER_HLL: &str = include_str!("../../shaders/science/plasma/euler_hll_f64.wgsl");

/// Mermin dielectric function ε(k,ω).
pub const WGSL_DIELECTRIC_MERMIN: &str =
    include_str!("../../shaders/science/plasma/dielectric_mermin_f64.wgsl");

/// Multi-component Mermin dielectric function.
pub const WGSL_DIELECTRIC_MULTICOMPONENT: &str =
    include_str!("../../shaders/science/plasma/dielectric_multicomponent_f64.wgsl");
