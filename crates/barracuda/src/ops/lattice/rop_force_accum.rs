// SPDX-License-Identifier: AGPL-3.0-or-later

//! ROP-accelerated fermion force accumulation (Tier 3 silicon routing).
//!
//! Uses `atomicAdd(i32)` in fixed-point to accumulate weighted force
//! contributions from multiple RHMC poles simultaneously, eliminating
//! `N_poles` sequential momentum-update dispatches.
//!
//! The fixed-point scale (2^20) provides ~6 significant digits — sufficient
//! for force accumulation where the Omelyan integrator error is O(dt^2).
//!
//! Absorbed from hotSpring V0632 (March 2026).
//!
//! ## Flow
//!
//! 1. Zero the i32 atomic accumulation buffer
//! 2. For each pole: dispatch fused force+atomicAdd shader (independent, no barriers)
//! 3. Single conversion dispatch: momentum += f64(accum) / scale
//!
//! ## Upstream Viability Assessment (August 2026)
//!
//! **Current path (compute atomicAdd)**: Proven, validated, already upstream.
//! Lights up ALU + L2 cache for scatter-add but ROPs remain dark.
//!
//! **Render-pass path (ROP additive blend)**: Prototype validated in hotSpring
//! (`render_force_accum.rs`). Uses `PointList` topology with `BlendOp::Add`
//! (src=One, dst=One) on `Rgba32Float` render target. Performance:
//! - RTX 3090: 7.8 G scatter-adds/s (~0.5× peak 112 ROPs)
//! - RX 6950 XT: 5.5 G scatter-adds/s (~0.05× peak 128 ROPs)
//!
//! **Viability**: HIGH for dynamical fermion HMC with many RHMC poles (N≥8).
//! LOW priority for pure gauge (no poles, force is single dispatch).
//!
//! **Upstream blockers**:
//! 1. barraCuda is currently compute-only (no render pipeline infrastructure)
//! 2. `Rgba32Float` blend on point primitives requires fragment shader stage
//! 3. Readback from render target → storage buffer adds a copy pass
//! 4. Fixed-point quantization in compute path is acceptable for O(dt²) integrator
//!
//! **Recommendation**: Keep compute atomic path as default. Upstream render path
//! when multi-pole RHMC campaigns demonstrate the atomic path as bottleneck.
//! hotSpring prototype (`render_force_accum.rs`) is the reference for integration.
//!
//! ## Fixed-Function Silicon — Capability-First Mapping
//!
//! Every fixed-function unit on the GPU was designed to solve a physics problem at
//! wire speed. Capability precedes use case — all units are exploration targets.
//!
//! | Unit | Hardware capability | QCD mapping | Status |
//! |------|--------------------|----|--------|
//! | **ROP** | Scatter-accumulate (additive blend) | Force accumulation across RHMC poles | MEASURED: 7.8G/s (3090) |
//! | **RT cores** | BVH spatial query O(log n) | Wilson loop tracing, parameter-space nearest-neighbor, multigrid coarsening on irregular geometries | MAPPED: 1.5 Mtri/s (3090) |
//! | **Tessellation** | Hardware h-refinement (subdivision) | Adaptive multigrid, non-uniform lattice generation, domain-adapted stencils near defects | THEORIZED |
//! | **Rasterizer** | Coverage/binning (primitive → fragment) | Domain decomposition, site→cell spatial sorting | MEASURED: 63 Msites/s |
//! | **Depth buffer** | Nearest-site lookup (z-test) | Voronoi coarsening, prolongation weights, smearing radius | MEASURED: 16 Mpx/s |
//! | **Video encoder** | Temporal coherence compression (NVENC) | Config archival (61:1), trajectory streaming, checkpoint delta | MEASURED: zero ALU contention |
//!
//! RT cores are not competitive for regular lattice neighbor lookup (O(1) index
//! arithmetic), but become relevant for: deformed lattices, adaptive meshes,
//! parameter-space hot-start queries, and Wilson loops on large geometries where
//! the path intersects O(L) links. See `bench_rt_core_probe.rs` and
//! `infra/whitePaper/subGen/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`.

use crate::device::capabilities::WORKGROUP_SIZE_1D;

/// Fixed-point scale factor for i32 atomic accumulation (2^20).
///
/// Provides ~6 decimal digits of precision, sufficient for force
/// accumulation where the Omelyan integrator error is O(dt^2).
pub const FIXED_POINT_SCALE: f64 = 1_048_576.0;

/// Number of f64 components per SU(3) link (3x3 complex = 18 reals).
pub const SU3_LINK_COMPONENTS: u32 = 18;

/// Build the uniform params buffer for one pole's fused force+accumulate dispatch.
///
/// Layout matches `su3_fermion_force_accumulate_rop_f64.wgsl` `Params` struct:
///   `volume: u32`, `pad0: u32`, `alpha_dt_hi: u32`, `alpha_dt_lo: u32`, `scale_factor: f64`
#[must_use]
pub fn make_pole_params(volume: u32, alpha_dt: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&volume.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    let bits = alpha_dt.to_bits();
    let hi = (bits >> 32) as u32;
    let lo = bits as u32;
    v.extend_from_slice(&hi.to_le_bytes());
    v.extend_from_slice(&lo.to_le_bytes());
    v.extend_from_slice(&FIXED_POINT_SCALE.to_le_bytes());
    v
}

/// Build the uniform params buffer for the final i32→f64 conversion dispatch.
///
/// Layout matches `su3_force_atomic_to_momentum_f64.wgsl` `Params` struct:
///   `n_values: u32`, `pad0: u32`, `inv_scale: f64`
#[must_use]
pub fn make_convert_params(n_values: u32) -> Vec<u8> {
    let inv_scale = 1.0 / FIXED_POINT_SCALE;
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&n_values.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&inv_scale.to_le_bytes());
    v
}

/// Calculate the number of atomic i32 entries for a given lattice volume.
///
/// Each site has 4 link directions, each link has 18 SU(3) components.
#[inline]
#[must_use]
pub const fn n_atomic_entries(volume: u32) -> u32 {
    volume * 4 * SU3_LINK_COMPONENTS
}

/// Workgroup count for the force accumulation shader (`workgroup_size=64`).
#[inline]
#[must_use]
pub const fn force_workgroups(volume: u32) -> u32 {
    volume.div_ceil(64)
}

/// Workgroup count for the conversion shader (`workgroup_size=256`).
#[inline]
#[must_use]
pub const fn convert_workgroups(n_values: u32) -> u32 {
    n_values.div_ceil(WORKGROUP_SIZE_1D)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pole_params_size() {
        let params = make_pole_params(1024, 0.01);
        assert_eq!(params.len(), 24);
    }

    #[test]
    fn convert_params_size() {
        let params = make_convert_params(1024);
        assert_eq!(params.len(), 16);
    }

    #[test]
    fn alpha_dt_round_trip() {
        let alpha_dt = std::f64::consts::PI * 0.001;
        let params = make_pole_params(100, alpha_dt);
        let hi = u32::from_le_bytes([params[8], params[9], params[10], params[11]]);
        let lo = u32::from_le_bytes([params[12], params[13], params[14], params[15]]);
        let recovered = f64::from_bits((u64::from(hi) << 32) | u64::from(lo));
        assert_eq!(recovered, alpha_dt);
    }

    #[test]
    fn n_entries_calculation() {
        assert_eq!(n_atomic_entries(1), 72);
        assert_eq!(n_atomic_entries(1024), 1024 * 4 * 18);
    }

    #[test]
    fn workgroup_counts() {
        assert_eq!(force_workgroups(64), 1);
        assert_eq!(force_workgroups(65), 2);
        assert_eq!(convert_workgroups(256), 1);
        assert_eq!(convert_workgroups(257), 2);
    }

    #[test]
    fn scale_precision_absolute() {
        let val = 0.5;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64 → i32 fixed-point conversion verified within test tolerance"
        )]
        let fixed = (val * FIXED_POINT_SCALE) as i32;
        let recovered = f64::from(fixed) / FIXED_POINT_SCALE;
        let abs_err = (recovered - val).abs();
        assert!(
            abs_err < 1e-6,
            "absolute error {abs_err} exceeds 10^-6 (2^-20 ≈ 10^-6)"
        );
    }

    #[test]
    fn scale_precision_force_range() {
        let val = 0.01;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64 → i32 fixed-point conversion verified within test tolerance"
        )]
        let fixed = (val * FIXED_POINT_SCALE) as i32;
        let recovered = f64::from(fixed) / FIXED_POINT_SCALE;
        let rel_err = ((recovered - val) / val).abs();
        assert!(
            rel_err < 1e-4,
            "relative error {rel_err} exceeds 10^-4 in force range"
        );
    }
}
