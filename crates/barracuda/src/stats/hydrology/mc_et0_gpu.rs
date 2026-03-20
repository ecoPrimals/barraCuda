// SPDX-License-Identifier: AGPL-3.0-or-later
//! Monte Carlo ET₀ uncertainty propagation on GPU.
//!
//! Perturbs FAO-56 meteorological inputs with normal noise and evaluates
//! Penman-Monteith ET₀ for each sample via the `mc_et0_propagate_f64` shader.

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use std::sync::Arc;

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
