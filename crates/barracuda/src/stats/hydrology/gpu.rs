// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated hydrological pipelines.
//!
//! Batch ET₀ computation, fused seasonal pipelines, and Monte Carlo
//! uncertainty propagation — all executed on GPU via WGSL shaders.

use super::{crop_coefficient, fao56_et0};
use crate::device::compute_pipeline::ComputeDispatch;
use std::sync::Arc;

// ── Hargreaves batch GPU ────────────────────────────────────────────────────

const SHADER_HARGREAVES: &str = include_str!("../../shaders/science/hargreaves_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HargreavesGpuParams {
    n_days: u32,
    _pad: [u32; 3],
}

pub struct HargreavesBatchGpu {
    device: Arc<crate::device::WgpuDevice>,
}

impl HargreavesBatchGpu {
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    pub fn dispatch(
        &self,
        ra: &[f64],
        t_max: &[f64],
        t_min: &[f64],
    ) -> crate::error::Result<Vec<f64>> {
        let n = ra.len();
        assert_eq!(n, t_max.len());
        assert_eq!(n, t_min.len());

        let ra_buf = self.device.create_buffer_f64_init("hargreaves:ra", ra);
        let tmax_buf = self.device.create_buffer_f64_init("hargreaves:tmax", t_max);
        let tmin_buf = self.device.create_buffer_f64_init("hargreaves:tmin", t_min);
        let out_buf = self.device.create_buffer_f64(n)?;
        let params = HargreavesGpuParams {
            n_days: n as u32,
            _pad: [0; 3],
        };
        let params_buf = self
            .device
            .create_uniform_buffer("hargreaves:params", &params);

        let wg = (n as u32).div_ceil(256);
        ComputeDispatch::new(&self.device, "hargreaves_batch")
            .shader(SHADER_HARGREAVES, "main")
            .f64()
            .storage_read(0, &ra_buf)
            .storage_read(1, &tmax_buf)
            .storage_read(2, &tmin_buf)
            .storage_rw(3, &out_buf)
            .uniform(4, &params_buf)
            .dispatch(wg, 1, 1)
            .submit();

        self.device.read_f64_buffer(&out_buf, n)
    }
}

// ── Fused seasonal pipeline GPU ─────────────────────────────────────────────

const SHADER_SEASONAL: &str = include_str!("../../shaders/science/seasonal_pipeline.wgsl");

/// GPU parameters for the fused seasonal pipeline.
///
/// Matches the `SeasonalParams` struct in `seasonal_pipeline.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SeasonalGpuParams {
    pub cell_count: u32,
    pub day_of_year: u32,
    pub stage_length: u32,
    pub day_in_stage: u32,
    pub kc_prev: f64,
    pub kc_next: f64,
    pub taw_default: f64,
    pub raw_fraction: f64,
    pub field_capacity: f64,
    _pad0: u32,
    _pad1: u32,
}

impl SeasonalGpuParams {
    /// Construct with all physical parameters; padding is set automatically.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cell_count: u32,
        day_of_year: u32,
        stage_length: u32,
        day_in_stage: u32,
        kc_prev: f64,
        kc_next: f64,
        taw_default: f64,
        raw_fraction: f64,
        field_capacity: f64,
    ) -> Self {
        Self {
            cell_count,
            day_of_year,
            stage_length,
            day_in_stage,
            kc_prev,
            kc_next,
            taw_default,
            raw_fraction,
            field_capacity,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Output from one cell of the seasonal pipeline.
#[derive(Debug, Clone, Copy)]
pub struct SeasonalOutput {
    pub et0: f64,
    pub kc: f64,
    pub etc: f64,
    pub theta_new: f64,
    pub stress: f64,
}

/// GPU executor for the fused seasonal pipeline.
///
/// Computes ET₀ → Kc → Water Balance → Yield stress in a single GPU dispatch
/// per spatial cell. Provenance: airSpring V035 → toadStool absorption.
///
/// # Input layout
///
/// `cell_weather`: 9 f64 per cell `[tmax, tmin, rh_max, rh_min, wind_2m, rs, elev, lat, soil_moisture_prev]`
///
/// # Output layout
///
/// 5 f64 per cell `[et0, kc, etc, theta_new, stress]`
pub struct SeasonalPipelineF64 {
    device: Arc<crate::device::WgpuDevice>,
}

impl SeasonalPipelineF64 {
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    pub fn dispatch(
        &self,
        cell_weather: &[f64],
        params: &SeasonalGpuParams,
    ) -> crate::error::Result<Vec<SeasonalOutput>> {
        let n = params.cell_count as usize;
        assert_eq!(
            cell_weather.len(),
            n * 9,
            "cell_weather must have 9 f64 per cell"
        );

        let weather_buf = self
            .device
            .create_buffer_f64_init("seasonal:weather", cell_weather);
        let out_buf = self.device.create_buffer_f64(n * 5)?;
        let params_buf = self.device.create_uniform_buffer("seasonal:params", params);

        ComputeDispatch::new(&self.device, "seasonal_pipeline")
            .shader(SHADER_SEASONAL, "seasonal_step")
            .f64()
            .storage_read(0, &weather_buf)
            .storage_rw(1, &out_buf)
            .uniform(2, &params_buf)
            .dispatch(n as u32, 1, 1)
            .submit();

        let raw = self.device.read_f64_buffer(&out_buf, n * 5)?;
        Ok(raw
            .chunks_exact(5)
            .map(|c| SeasonalOutput {
                et0: c[0],
                kc: c[1],
                etc: c[2],
                theta_new: c[3],
                stress: c[4],
            })
            .collect())
    }

    /// CPU reference implementation for validation.
    pub fn execute_cpu(
        cell_weather: &[f64],
        kc_prev: f64,
        kc_next: f64,
        day_in_stage: u32,
        stage_length: u32,
        taw_default: f64,
        raw_fraction: f64,
        field_capacity: f64,
        doy: u32,
    ) -> Vec<SeasonalOutput> {
        let n = cell_weather.len() / 9;
        (0..n)
            .map(|i| {
                let base = i * 9;
                let et0 = fao56_et0(
                    cell_weather[base],
                    cell_weather[base + 1],
                    cell_weather[base + 2],
                    cell_weather[base + 3],
                    cell_weather[base + 4],
                    cell_weather[base + 5],
                    cell_weather[base + 6],
                    cell_weather[base + 7],
                    doy,
                )
                .unwrap_or(0.0);
                let kc = crop_coefficient(kc_prev, kc_next, day_in_stage, stage_length);
                let etc = et0 * kc;
                let theta_prev = cell_weather[base + 8];
                let raw = taw_default * raw_fraction;
                let ks = if theta_prev > raw {
                    ((taw_default - theta_prev) / (taw_default - raw).max(0.001)).max(0.0)
                } else {
                    1.0
                };
                let etc_adj = ks * etc;
                let theta_new = (theta_prev - etc_adj).clamp(0.0, field_capacity);
                let stress = 1.0 - ks;
                SeasonalOutput {
                    et0,
                    kc,
                    etc,
                    theta_new,
                    stress,
                }
            })
            .collect()
    }
}

// ── Monte Carlo ET₀ uncertainty propagation (GPU) ───────────────────────────

const SHADER_MC_ET0: &str = include_str!("../../shaders/bio/mc_et0_propagate_f64.wgsl");

/// Base meteorological inputs for a single site-day (9 f64).
#[derive(Debug, Clone, Copy)]
pub struct Fao56BaseInputs {
    pub t_max: f64,
    pub t_min: f64,
    pub rh_max: f64,
    pub rh_min: f64,
    pub wind_kmh: f64,
    pub sun_hours: f64,
    pub latitude: f64,
    pub altitude: f64,
    pub day_of_year: f64,
}

/// Uncertainty (σ) for each perturbed input.
#[derive(Debug, Clone, Copy)]
pub struct Fao56Uncertainties {
    pub sigma_t_max: f64,
    pub sigma_t_min: f64,
    pub sigma_rh_max: f64,
    pub sigma_rh_min: f64,
    pub sigma_wind_frac: f64,
    pub sigma_sun_frac: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct McEt0Params {
    n_samples: u32,
    _pad: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Monte Carlo uncertainty propagation through FAO-56 ET₀ on GPU.
///
/// Generates `n_samples` perturbed ET₀ values using Box-Muller normal noise
/// and xoshiro128** PRNG. Each sample independently perturbs meteorological
/// inputs and evaluates the full Penman-Monteith equation chain.
///
/// Provenance: groundSpring V10 handoff → toadStool absorption.
pub struct McEt0PropagateGpu {
    device: Arc<crate::device::WgpuDevice>,
}

impl McEt0PropagateGpu {
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch Monte Carlo ET₀ propagation.
    ///
    /// Returns `n_samples` ET₀ values drawn from the uncertainty distribution.
    pub fn dispatch(
        &self,
        base: &Fao56BaseInputs,
        uncert: &Fao56Uncertainties,
        n_samples: u32,
    ) -> crate::error::Result<Vec<f64>> {
        let base_data = [
            base.t_max,
            base.t_min,
            base.rh_max,
            base.rh_min,
            base.wind_kmh,
            base.sun_hours,
            base.latitude,
            base.altitude,
            base.day_of_year,
        ];
        let uncert_data = [
            uncert.sigma_t_max,
            uncert.sigma_t_min,
            uncert.sigma_rh_max,
            uncert.sigma_rh_min,
            uncert.sigma_wind_frac,
            uncert.sigma_sun_frac,
        ];

        let params = McEt0Params {
            n_samples,
            _pad: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buf = self.device.create_uniform_buffer("mc_et0:params", &params);
        let base_buf = self
            .device
            .create_buffer_f64_init("mc_et0:base", &base_data);
        let uncert_buf = self
            .device
            .create_buffer_f64_init("mc_et0:uncert", &uncert_data);

        let seed_count = n_samples as usize * 4;
        let seeds: Vec<u32> = (0..seed_count as u32)
            .map(|i| i.wrapping_mul(2654435761).wrapping_add(1))
            .collect();
        let seeds_buf = self.device.create_buffer_u32_init("mc_et0:seeds", &seeds);
        let out_buf = self.device.create_buffer_f64(n_samples as usize)?;

        let wg = n_samples.div_ceil(64);
        ComputeDispatch::new(&self.device, "mc_et0_propagate")
            .shader(SHADER_MC_ET0, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &base_buf)
            .storage_read(2, &uncert_buf)
            .storage_rw(3, &seeds_buf)
            .storage_rw(4, &out_buf)
            .dispatch(wg, 1, 1)
            .submit();

        self.device.read_f64_buffer(&out_buf, n_samples as usize)
    }
}
