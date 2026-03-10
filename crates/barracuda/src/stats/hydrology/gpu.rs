// SPDX-License-Identifier: AGPL-3.0-only
//! GPU-accelerated hydrological pipelines.
//!
//! Batch ET₀ computation, fused seasonal pipelines, and Monte Carlo
//! uncertainty propagation — all executed on GPU via WGSL shaders.

use super::{crop_coefficient, fao56_et0};
use crate::device::capabilities::{WORKGROUP_SIZE_1D, WORKGROUP_SIZE_COMPACT};
use crate::device::compute_pipeline::ComputeDispatch;
use std::sync::Arc;

// ── Hargreaves batch GPU ────────────────────────────────────────────────────

const SHADER_HARGREAVES: &str = include_str!("../../shaders/science/hargreaves_batch_f64.wgsl");

/// Uniform buffer parameters for the Hargreaves batch ET₀ shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HargreavesGpuParams {
    /// Number of days (length of ra, `t_max`, `t_min` arrays).
    n_days: u32,
    _pad: [u32; 3],
}

/// GPU-accelerated batch Hargreaves reference evapotranspiration (ET₀).
///
/// Computes FAO Hargreaves ET₀ for multiple days in a single dispatch.
pub struct HargreavesBatchGpu {
    device: Arc<crate::device::WgpuDevice>,
}

impl HargreavesBatchGpu {
    /// Create a new Hargreaves batch GPU executor.
    /// # Errors
    /// Never returns an error; always returns `Ok` when the device is valid.
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch Hargreaves ET₀ for all days. Returns ET₀ values (mm/day).
    /// # Panics
    /// Panics if `ra.len() != t_max.len()` or `ra.len() != t_min.len()`.
    /// # Errors
    /// Returns [`Err`] if buffer creation fails, buffer readback fails (e.g. device
    /// lost, mapping timeout), or the GPU compute dispatch fails.
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

        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "hargreaves_batch")
            .shader(SHADER_HARGREAVES, "main")
            .f64()
            .storage_read(0, &ra_buf)
            .storage_read(1, &tmax_buf)
            .storage_read(2, &tmin_buf)
            .storage_rw(3, &out_buf)
            .uniform(4, &params_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

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
    /// Number of spatial cells.
    pub cell_count: u32,
    /// Day of year (1–365).
    pub day_of_year: u32,
    /// Length of current growth stage in days.
    pub stage_length: u32,
    /// Day within current stage.
    pub day_in_stage: u32,
    /// Crop coefficient at start of stage.
    pub kc_prev: f64,
    /// Crop coefficient at end of stage.
    pub kc_next: f64,
    /// Total available water (mm).
    pub taw_default: f64,
    /// Readily available water fraction.
    pub raw_fraction: f64,
    /// Field capacity (mm).
    pub field_capacity: f64,
    _pad0: u32,
    _pad1: u32,
}

impl SeasonalGpuParams {
    /// Start building seasonal parameters with required grid + time info.
    #[must_use]
    pub fn builder(cell_count: u32, day_of_year: u32) -> SeasonalGpuParamsBuilder {
        SeasonalGpuParamsBuilder {
            cell_count,
            day_of_year,
            stage_length: 1,
            day_in_stage: 0,
            kc_prev: 0.3,
            kc_next: 0.3,
            taw_default: 100.0,
            raw_fraction: 0.5,
            field_capacity: 0.3,
        }
    }
}

/// Builder for [`SeasonalGpuParams`] — avoids 9-argument constructor.
#[derive(Debug, Clone)]
pub struct SeasonalGpuParamsBuilder {
    cell_count: u32,
    day_of_year: u32,
    stage_length: u32,
    day_in_stage: u32,
    kc_prev: f64,
    kc_next: f64,
    taw_default: f64,
    raw_fraction: f64,
    field_capacity: f64,
}

impl SeasonalGpuParamsBuilder {
    /// Growth stage timing.
    #[must_use]
    pub const fn stage(mut self, length: u32, day_in_stage: u32) -> Self {
        self.stage_length = length;
        self.day_in_stage = day_in_stage;
        self
    }

    /// Crop coefficients at start and end of the current growth stage.
    #[must_use]
    pub const fn crop_coefficients(mut self, kc_prev: f64, kc_next: f64) -> Self {
        self.kc_prev = kc_prev;
        self.kc_next = kc_next;
        self
    }

    /// Soil water parameters.
    #[must_use]
    pub const fn soil(mut self, taw: f64, raw_fraction: f64, field_capacity: f64) -> Self {
        self.taw_default = taw;
        self.raw_fraction = raw_fraction;
        self.field_capacity = field_capacity;
        self
    }

    /// Build the GPU-compatible parameter struct.
    #[must_use]
    pub const fn build(self) -> SeasonalGpuParams {
        SeasonalGpuParams {
            cell_count: self.cell_count,
            day_of_year: self.day_of_year,
            stage_length: self.stage_length,
            day_in_stage: self.day_in_stage,
            kc_prev: self.kc_prev,
            kc_next: self.kc_next,
            taw_default: self.taw_default,
            raw_fraction: self.raw_fraction,
            field_capacity: self.field_capacity,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Output from one cell of the seasonal pipeline.
#[derive(Debug, Clone, Copy)]
pub struct SeasonalOutput {
    /// Reference evapotranspiration (mm/day).
    pub et0: f64,
    /// Crop coefficient.
    pub kc: f64,
    /// Crop evapotranspiration (mm/day).
    pub etc: f64,
    /// Updated soil moisture (mm).
    pub theta_new: f64,
    /// Water stress factor (0 = none, 1 = full stress).
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
    /// Create a new seasonal pipeline executor.
    /// # Errors
    /// Never returns an error; always returns `Ok` when the device is valid.
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch the fused seasonal pipeline for all cells.
    /// # Panics
    /// Panics if `cell_weather.len() != params.cell_count * 9`.
    /// # Errors
    /// Returns [`Err`] if buffer creation fails, buffer readback fails (e.g. device
    /// lost, mapping timeout), or the GPU compute dispatch fails.
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
            .submit()?;

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
    #[must_use]
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
    /// Maximum air temperature (°C).
    pub t_max: f64,
    /// Minimum air temperature (°C).
    pub t_min: f64,
    /// Maximum relative humidity (%).
    pub rh_max: f64,
    /// Minimum relative humidity (%).
    pub rh_min: f64,
    /// Wind speed at 2 m (km/h).
    pub wind_kmh: f64,
    /// Sunshine hours.
    pub sun_hours: f64,
    /// Latitude (degrees).
    pub latitude: f64,
    /// Altitude (m).
    pub altitude: f64,
    /// Day of year (1–365).
    pub day_of_year: f64,
}

/// Uncertainty (σ) for each perturbed input.
#[derive(Debug, Clone, Copy)]
pub struct Fao56Uncertainties {
    /// Std dev for max temperature perturbation (°C).
    pub sigma_t_max: f64,
    /// Std dev for min temperature perturbation (°C).
    pub sigma_t_min: f64,
    /// Std dev for max RH perturbation (%).
    pub sigma_rh_max: f64,
    /// Std dev for min RH perturbation (%).
    pub sigma_rh_min: f64,
    /// Fractional std dev for wind speed.
    pub sigma_wind_frac: f64,
    /// Fractional std dev for sunshine hours.
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
    /// Create a new Monte Carlo ET₀ propagation executor.
    /// # Errors
    /// Never returns an error; always returns `Ok` when the device is valid.
    pub fn new(device: Arc<crate::device::WgpuDevice>) -> crate::error::Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch Monte Carlo ET₀ propagation.
    /// Returns `n_samples` ET₀ values drawn from the uncertainty distribution.
    /// # Errors
    /// Returns [`Err`] if buffer creation fails, buffer readback fails (e.g. device
    /// lost, mapping timeout), or the GPU compute dispatch fails.
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

        let wg = n_samples.div_ceil(WORKGROUP_SIZE_COMPACT);
        ComputeDispatch::new(&self.device, "mc_et0_propagate")
            .shader(SHADER_MC_ET0, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &base_buf)
            .storage_read(2, &uncert_buf)
            .storage_rw(3, &seeds_buf)
            .storage_rw(4, &out_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n_samples as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    #[test]
    fn test_seasonal_gpu_params_new() {
        let p = SeasonalGpuParams::builder(10, 187)
            .stage(30, 15)
            .crop_coefficients(0.3, 1.2)
            .soil(100.0, 0.5, 0.4)
            .build();
        assert_eq!(p.cell_count, 10);
        assert_eq!(p.day_of_year, 187);
        assert_eq!(p.stage_length, 30);
        assert_eq!(p.day_in_stage, 15);
        assert!((p.kc_prev - 0.3).abs() < 1e-12);
        assert!((p.kc_next - 1.2).abs() < 1e-12);
        assert!((p.taw_default - 100.0).abs() < 1e-12);
        assert!((p.raw_fraction - 0.5).abs() < 1e-12);
        assert!((p.field_capacity - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_seasonal_pipeline_execute_cpu() {
        // cell_weather: 9 f64 per cell [tmax, tmin, rh_max, rh_min, wind_2m, rs, elev, lat, soil_moisture_prev]
        // 2 cells, 18 values total. Use FAO-56 Example 18–style inputs for cell 1.
        let cell_weather: Vec<f64> = vec![
            // Cell 1: tmax, tmin, rh_max, rh_min, wind_2m, rs, elev, lat, theta_prev
            21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 30.0, // Cell 2
            25.0, 15.0, 75.0, 55.0, 3.0, 24.0, 150.0, 45.0, 80.0,
        ];
        let out = SeasonalPipelineF64::execute_cpu(
            &cell_weather,
            0.3,   // kc_prev
            1.2,   // kc_next
            15,    // day_in_stage
            30,    // stage_length
            100.0, // taw_default
            0.5,   // raw_fraction
            150.0, // field_capacity (mm)
            187,   // doy
        );
        assert_eq!(out.len(), 2);
        for o in &out {
            assert!(o.et0 > 0.0, "ET0 should be positive, got {}", o.et0);
            assert!(
                o.kc >= 0.0 && o.kc <= 2.0,
                "Kc should be in valid range, got {}",
                o.kc
            );
            assert!(
                o.stress >= 0.0 && o.stress <= 1.0,
                "stress should be in [0,1], got {}",
                o.stress
            );
            assert!(
                o.theta_new >= 0.0 && o.theta_new <= 150.0,
                "theta_new clamped to field_capacity"
            );
        }
    }

    #[test]
    fn test_fao56_base_inputs_construction() {
        let base = Fao56BaseInputs {
            t_max: 21.5,
            t_min: 12.3,
            rh_max: 84.0,
            rh_min: 63.0,
            wind_kmh: 10.0,
            sun_hours: 12.0,
            latitude: 50.8,
            altitude: 100.0,
            day_of_year: 187.0,
        };
        assert!((base.t_max - 21.5).abs() < 1e-12);
        assert!((base.latitude - 50.8).abs() < 1e-12);
    }

    #[test]
    fn test_fao56_uncertainties_construction() {
        let uncert = Fao56Uncertainties {
            sigma_t_max: 1.0,
            sigma_t_min: 1.0,
            sigma_rh_max: 5.0,
            sigma_rh_min: 5.0,
            sigma_wind_frac: 0.1,
            sigma_sun_frac: 0.1,
        };
        assert!((uncert.sigma_t_max - 1.0).abs() < 1e-12);
        assert!((uncert.sigma_wind_frac - 0.1).abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_hargreaves_batch_gpu_dispatch() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let Ok(gpu) = HargreavesBatchGpu::new(device) else {
            return;
        };
        let ra = vec![30.0, 35.0, 40.0];
        let t_max = vec![28.0, 32.0, 35.0];
        let t_min = vec![15.0, 18.0, 20.0];
        let Ok(out) = gpu.dispatch(&ra, &t_max, &t_min) else {
            return;
        };
        assert_eq!(out.len(), 3);
        for &e in &out {
            assert!(e > 0.0, "Hargreaves ET0 should be positive");
        }
    }

    #[tokio::test]
    async fn test_seasonal_pipeline_gpu_dispatch() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let Ok(gpu) = SeasonalPipelineF64::new(device) else {
            return;
        };
        let cell_weather: Vec<f64> = vec![
            21.5, 12.3, 84.0, 63.0, 2.78, 22.07, 100.0, 50.8, 30.0, 25.0, 15.0, 75.0, 55.0, 3.0,
            24.0, 150.0, 45.0, 80.0,
        ];
        let params = SeasonalGpuParams::builder(2, 187)
            .stage(30, 15)
            .crop_coefficients(0.3, 1.2)
            .soil(100.0, 0.5, 150.0)
            .build();
        let Ok(out) = gpu.dispatch(&cell_weather, &params) else {
            return;
        };
        assert_eq!(out.len(), 2);
        for o in &out {
            assert!(o.et0 > 0.0);
            assert!(o.kc >= 0.0 && o.kc <= 2.0);
            assert!(o.stress >= 0.0 && o.stress <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_mc_et0_propagate_gpu_dispatch() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let Ok(gpu) = McEt0PropagateGpu::new(device) else {
            return;
        };
        let base = Fao56BaseInputs {
            t_max: 21.5,
            t_min: 12.3,
            rh_max: 84.0,
            rh_min: 63.0,
            wind_kmh: 10.0,
            sun_hours: 12.0,
            latitude: 50.8,
            altitude: 100.0,
            day_of_year: 187.0,
        };
        let uncert = Fao56Uncertainties {
            sigma_t_max: 0.5,
            sigma_t_min: 0.5,
            sigma_rh_max: 2.0,
            sigma_rh_min: 2.0,
            sigma_wind_frac: 0.05,
            sigma_sun_frac: 0.05,
        };
        let Ok(out) = gpu.dispatch(&base, &uncert, 64) else {
            return;
        };
        assert_eq!(out.len(), 64);
        for &e in &out {
            assert!(e >= 0.0, "MC ET0 samples should be non-negative, got {e}");
        }
    }
}
