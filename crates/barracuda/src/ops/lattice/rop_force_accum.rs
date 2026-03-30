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
    n_values.div_ceil(256)
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
        #[expect(clippy::cast_possible_truncation)]
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
        #[expect(clippy::cast_possible_truncation)]
        let fixed = (val * FIXED_POINT_SCALE) as i32;
        let recovered = f64::from(fixed) / FIXED_POINT_SCALE;
        let rel_err = ((recovered - val) / val).abs();
        assert!(
            rel_err < 1e-4,
            "relative error {rel_err} exceeds 10^-4 in force range"
        );
    }
}
