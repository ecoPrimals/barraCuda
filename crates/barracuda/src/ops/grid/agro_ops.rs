// SPDX-License-Identifier: AGPL-3.0-only
//
//! Agricultural/environmental computing primitives (airSpring absorption).
//!
//! - **Hargreaves ET₀**: FAO reference evapotranspiration
//! - **Dual Kc**: FAO dual crop coefficient `ETc`
//! - **Van Genuchten**: Soil moisture retention θ(h) and K(θ)
//! - **Batched crop pipeline**: Water balance over time steps

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_HARGREAVES_ET0: &str = include_str!("../../shaders/grid/hargreaves_et0_f64.wgsl");
const SHADER_DUAL_KC: &str = include_str!("../../shaders/grid/dual_kc_f64.wgsl");
const SHADER_VAN_GENUCHTEN: &str = include_str!("../../shaders/grid/van_genuchten_f64.wgsl");
const SHADER_BATCHED_CROP_PIPELINE: &str =
    include_str!("../../shaders/grid/batched_crop_pipeline_f64.wgsl");

// ── Hargreaves ET₀ ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HargreavesParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// FAO Hargreaves reference evapotranspiration: ET₀ = 0.0023 * Ra * (`T_mean` + 17.8) * (`T_max` - `T_min)^0.5`
///
/// # Panics
/// Panics if `t_min.len() != t_max.len()` or `ra.len() != t_max.len()`.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn hargreaves_et0(
    device: &Arc<WgpuDevice>,
    t_max: &[f64],
    t_min: &[f64],
    ra: &[f64],
) -> Result<Vec<f64>> {
    let n = t_max.len();
    assert_eq!(t_min.len(), n, "t_min length must match t_max");
    assert_eq!(ra.len(), n, "ra length must match t_max");

    let t_max_buf = device.create_buffer_f64_init("hargreaves:t_max", t_max);
    let t_min_buf = device.create_buffer_f64_init("hargreaves:t_min", t_min);
    let ra_buf = device.create_buffer_f64_init("hargreaves:ra", ra);
    let out_buf = device.create_buffer_f64(n)?;
    let params = HargreavesParams {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_uniform_buffer("hargreaves:params", &params);

    ComputeDispatch::new(device, "hargreaves_et0")
        .shader(SHADER_HARGREAVES_ET0, "main")
        .f64()
        .storage_read(0, &t_max_buf)
        .storage_read(1, &t_min_buf)
        .storage_read(2, &ra_buf)
        .storage_rw(3, &out_buf)
        .uniform(4, &params_buf)
        .dispatch_1d(n as u32)
        .submit()?;

    device.read_f64_buffer(&out_buf, n)
}

// ── Dual Kc ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DualKcParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// FAO dual crop coefficient: `ETc` = (Kcb * Ks + Ke) * ET₀
///
/// # Panics
/// Panics if `ks.len()`, `ke.len()`, or `et0.len()` != `kcb.len()`.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn dual_kc(
    device: &Arc<WgpuDevice>,
    kcb: &[f64],
    ks: &[f64],
    ke: &[f64],
    et0: &[f64],
) -> Result<Vec<f64>> {
    let n = kcb.len();
    assert_eq!(ks.len(), n, "ks length must match kcb");
    assert_eq!(ke.len(), n, "ke length must match kcb");
    assert_eq!(et0.len(), n, "et0 length must match kcb");

    let kcb_buf = device.create_buffer_f64_init("dual_kc:kcb", kcb);
    let ks_buf = device.create_buffer_f64_init("dual_kc:ks", ks);
    let ke_buf = device.create_buffer_f64_init("dual_kc:ke", ke);
    let et0_buf = device.create_buffer_f64_init("dual_kc:et0", et0);
    let out_buf = device.create_buffer_f64(n)?;
    let params = DualKcParams {
        n: n as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_uniform_buffer("dual_kc:params", &params);

    ComputeDispatch::new(device, "dual_kc")
        .shader(SHADER_DUAL_KC, "main")
        .f64()
        .storage_read(0, &kcb_buf)
        .storage_read(1, &ks_buf)
        .storage_read(2, &ke_buf)
        .storage_read(3, &et0_buf)
        .storage_rw(4, &out_buf)
        .uniform(5, &params_buf)
        .dispatch_1d(n as u32)
        .submit()?;

    device.read_f64_buffer(&out_buf, n)
}

// ── Van Genuchten ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VanGenuchtenParams {
    n: u32,
    _pad: u32,
    theta_r: f64,
    theta_s: f64,
    alpha_vg: f64,
    n_vg: f64,
    k_s: f64,
    l_vg: f64,
}

/// Van Genuchten soil moisture retention θ(h) and hydraulic conductivity K(θ).
///
/// Returns `(theta, k)` where theta is volumetric water content and k is conductivity.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn van_genuchten(
    device: &Arc<WgpuDevice>,
    h: &[f64],
    theta_r: f64,
    theta_s: f64,
    alpha_vg: f64,
    n_vg: f64,
    k_s: f64,
    l_vg: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = h.len();

    let h_buf = device.create_buffer_f64_init("van_genuchten:h", h);
    let out_theta_buf = device.create_buffer_f64(n)?;
    let out_k_buf = device.create_buffer_f64(n)?;
    let params = VanGenuchtenParams {
        n: n as u32,
        _pad: 0,
        theta_r,
        theta_s,
        alpha_vg,
        n_vg,
        k_s,
        l_vg,
    };
    let params_buf = device.create_uniform_buffer("van_genuchten:params", &params);

    ComputeDispatch::new(device, "van_genuchten")
        .shader(SHADER_VAN_GENUCHTEN, "main")
        .f64()
        .storage_read(0, &h_buf)
        .storage_rw(1, &out_theta_buf)
        .storage_rw(2, &out_k_buf)
        .uniform(3, &params_buf)
        .dispatch_1d(n as u32)
        .submit()?;

    let theta = device.read_f64_buffer(&out_theta_buf, n)?;
    let k = device.read_f64_buffer(&out_k_buf, n)?;
    Ok((theta, k))
}

// ── Batched Crop Pipeline ───────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BatchedCropParams {
    n_steps: u32,
    n_cells: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Combined crop water balance over time steps.
///
/// For each cell: `soil_water`[t+1] = `soil_water`[t] + precip[t] - `ETc`[t] - drainage[t]
/// with drainage = max(0, `soil_water` - `field_capacity`).
///
/// - `precip`, `etc_vals`: [`n_cells` * `n_steps`], index = cell * `n_steps` + step
/// - `field_capacity`: [`n_cells`]
/// - `soil_water`: [`n_cells`] initial values (overwritten with final values)
/// - Returns `(soil_water, drainage)` where drainage is [`n_cells` * `n_steps`]
///
/// # Panics
/// Panics if `precip.len()`, `etc_vals.len()`, or `field_capacity.len()` do not match expected dimensions.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, buffer readback/mapping fails, or the device is lost.
pub fn batched_crop_pipeline(
    device: &Arc<WgpuDevice>,
    precip: &[f64],
    etc_vals: &[f64],
    field_capacity: &[f64],
    soil_water: &[f64],
    n_steps: u32,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n_cells = soil_water.len();
    let total = n_cells * n_steps as usize;
    assert_eq!(precip.len(), total, "precip must be n_cells * n_steps");
    assert_eq!(etc_vals.len(), total, "etc_vals must be n_cells * n_steps");
    assert_eq!(
        field_capacity.len(),
        n_cells,
        "field_capacity must be n_cells"
    );

    let precip_buf = device.create_buffer_f64_init("batched_crop:precip", precip);
    let etc_buf = device.create_buffer_f64_init("batched_crop:etc_vals", etc_vals);
    let fc_buf = device.create_buffer_f64_init("batched_crop:field_capacity", field_capacity);
    let sw_buf = device.create_buffer_f64_init("batched_crop:soil_water", soil_water);
    let drain_buf = device.create_buffer_f64(total)?;
    let params = BatchedCropParams {
        n_steps,
        n_cells: n_cells as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_uniform_buffer("batched_crop:params", &params);

    ComputeDispatch::new(device, "batched_crop_pipeline")
        .shader(SHADER_BATCHED_CROP_PIPELINE, "main")
        .f64()
        .storage_read(0, &precip_buf)
        .storage_read(1, &etc_buf)
        .storage_read(2, &fc_buf)
        .storage_rw(3, &sw_buf)
        .storage_rw(4, &drain_buf)
        .uniform(5, &params_buf)
        .dispatch_1d(n_cells as u32)
        .submit()?;

    let soil_water_out = device.read_f64_buffer(&sw_buf, n_cells)?;
    let drainage = device.read_f64_buffer(&drain_buf, total)?;
    Ok((soil_water_out, drainage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hargreaves_params_layout() {
        // repr(C): n(u32) + _pad0(u32) + _pad1(u32) + _pad2(u32) = 16 bytes
        let params = HargreavesParams {
            n: 100,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        assert_eq!(std::mem::size_of::<HargreavesParams>(), 16);
        assert_eq!(std::mem::align_of::<HargreavesParams>(), 4);
        assert_eq!(params.n, 100);
    }

    #[test]
    fn test_dual_kc_params_layout() {
        // repr(C): n(u32) + _pad0(u32) + _pad1(u32) + _pad2(u32) = 16 bytes
        let params = DualKcParams {
            n: 64,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        assert_eq!(std::mem::size_of::<DualKcParams>(), 16);
        assert_eq!(std::mem::align_of::<DualKcParams>(), 4);
        assert_eq!(params.n, 64);
    }

    #[test]
    fn test_van_genuchten_params_layout() {
        // repr(C): n(u32) + _pad(u32) + 6×f64 = 8 + 48 = 56 bytes
        let params = VanGenuchtenParams {
            n: 10,
            _pad: 0,
            theta_r: 0.05,
            theta_s: 0.4,
            alpha_vg: 0.01,
            n_vg: 1.5,
            k_s: 1e-5,
            l_vg: 0.5,
        };
        assert_eq!(std::mem::size_of::<VanGenuchtenParams>(), 56);
        assert_eq!(std::mem::align_of::<VanGenuchtenParams>(), 8);
        assert_eq!(params.n, 10);
    }

    #[test]
    fn test_crop_pipeline_params_layout() {
        // repr(C): n_steps(u32) + n_cells(u32) + _pad0(u32) + _pad1(u32) = 16 bytes
        let params = BatchedCropParams {
            n_steps: 365,
            n_cells: 1000,
            _pad0: 0,
            _pad1: 0,
        };
        assert_eq!(std::mem::size_of::<BatchedCropParams>(), 16);
        assert_eq!(std::mem::align_of::<BatchedCropParams>(), 4);
        assert_eq!(params.n_steps, 365);
    }

    #[test]
    fn test_shader_sources_valid() {
        for (name, shader) in [
            ("hargreaves_et0", SHADER_HARGREAVES_ET0),
            ("dual_kc", SHADER_DUAL_KC),
            ("van_genuchten", SHADER_VAN_GENUCHTEN),
            ("batched_crop_pipeline", SHADER_BATCHED_CROP_PIPELINE),
        ] {
            assert!(!shader.is_empty(), "{name} shader must not be empty");
            assert!(
                shader.contains("fn main"),
                "{name} shader must contain 'fn main'"
            );
        }
    }
}
