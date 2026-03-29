// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Hill dose-response with Emax model.
//!
//! E(c) = Emax * c^n / (c^n + EC50^n)
//!
//! Absorbed from healthSpring V44 (March 2026).

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/health/hill_dose_response_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HillParams {
    emax: f64,
    ec50: f64,
    hill_n: f64,
    n_concs: u32,
    _pad: u32,
}

/// Hill dose-response configuration.
#[derive(Debug, Clone, Copy)]
pub struct HillConfig {
    /// Maximum effect (Emax).
    pub emax: f64,
    /// Concentration at half-maximal effect (EC50).
    pub ec50: f64,
    /// Hill coefficient (steepness).
    pub hill_n: f64,
}

/// GPU-accelerated Hill dose-response computation.
pub struct HillDoseResponseGpu {
    device: Arc<WgpuDevice>,
}

impl HillDoseResponseGpu {
    /// Create a new `HillDoseResponseGpu` for the given device.
    #[must_use]
    pub const fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute dose-response for a batch of concentrations.
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline creation, dispatch, or readback failure.
    pub fn compute(&self, concentrations: &[f64], config: &HillConfig) -> Result<Vec<f64>> {
        let n = concentrations.len();
        let params = HillParams {
            emax: config.emax,
            ec50: config.ec50,
            hill_n: config.hill_n,
            n_concs: n as u32,
            _pad: 0,
        };

        let in_buf = self
            .device
            .create_buffer_f64_init("hill:input", concentrations);
        let out_buf = self.device.create_buffer_f64(n)?;
        let params_buf = self.device.create_uniform_buffer("hill:params", &params);

        let wg_count = (n as u32).div_ceil(256);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "hill_dose_response")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &in_buf)
            .storage_rw(1, &out_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n)
    }
}

/// CPU reference implementation of Hill dose-response.
///
/// Used for GPU parity validation and fallback when no device is available.
#[must_use]
pub fn hill_dose_response_cpu(concentration: f64, config: &HillConfig) -> f64 {
    if concentration <= 0.0 {
        return 0.0;
    }
    let c_n = concentration.powf(config.hill_n);
    let ec50_n = config.ec50.powf(config.hill_n);
    config.emax * c_n / (c_n + ec50_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<HillParams>(), 32);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("hill"));
        assert!(SHADER.contains("power_f64"));
        assert!(SHADER.contains("Params"));
    }

    #[test]
    fn cpu_reference_basic() {
        let config = HillConfig {
            emax: 100.0,
            ec50: 10.0,
            hill_n: 1.0,
        };
        let at_ec50 = hill_dose_response_cpu(10.0, &config);
        assert!((at_ec50 - 50.0).abs() < 1e-10, "E(EC50) should be Emax/2");
    }

    #[test]
    fn cpu_reference_zero_concentration() {
        let config = HillConfig {
            emax: 100.0,
            ec50: 10.0,
            hill_n: 2.0,
        };
        assert_eq!(hill_dose_response_cpu(0.0, &config), 0.0);
    }

    #[test]
    fn cpu_reference_steep_hill() {
        let config = HillConfig {
            emax: 1.0,
            ec50: 5.0,
            hill_n: 4.0,
        };
        let low = hill_dose_response_cpu(2.5, &config);
        let high = hill_dose_response_cpu(10.0, &config);
        assert!(low < 0.1, "far below EC50 with steep n should give low E");
        assert!(high > 0.9, "far above EC50 with steep n should give high E");
    }
}
