// SPDX-License-Identifier: AGPL-3.0-only

//! GPU batch Michaelis-Menten PK simulation.
//!
//! Each thread simulates one patient with per-patient Vmax variation.
//! Absorbed from healthSpring V19 (Exp083, Exp085).

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/health/michaelis_menten_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MmParams {
    vmax: f64,
    km: f64,
    vd: f64,
    dt: f64,
    n_steps: u32,
    n_patients: u32,
    base_seed: u32,
    _pad: u32,
}

/// Michaelis-Menten batch GPU simulation configuration.
#[derive(Debug, Clone, Copy)]
pub struct MmBatchConfig {
    /// Maximum elimination rate (mg/L/h).
    pub vmax: f64,
    /// Michaelis constant (mg/L).
    pub km: f64,
    /// Volume of distribution (L).
    pub vd: f64,
    /// Integration time step (h).
    pub dt: f64,
    /// Number of Euler steps.
    pub n_steps: u32,
    /// Number of patients (GPU threads).
    pub n_patients: u32,
    /// PRNG base seed.
    pub seed: u32,
}

/// GPU-accelerated batch Michaelis-Menten PK simulation.
pub struct MichaelisMentenBatchGpu {
    device: Arc<WgpuDevice>,
}

impl MichaelisMentenBatchGpu {
    /// Create a new `MichaelisMentenBatchGpu` for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Run batch PK simulation, returning per-patient AUC values.
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline creation, dispatch, or readback failure.
    pub fn simulate(&self, config: &MmBatchConfig) -> Result<Vec<f64>> {
        let params = MmParams {
            vmax: config.vmax,
            km: config.km,
            vd: config.vd,
            dt: config.dt,
            n_steps: config.n_steps,
            n_patients: config.n_patients,
            base_seed: config.seed,
            _pad: 0,
        };

        let n = config.n_patients as usize;
        let out_buf = self.device.create_buffer_f64(n)?;
        let params_buf = self
            .device
            .create_uniform_buffer("mm_batch:params", &params);

        let wg_count = config.n_patients.div_ceil(256);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "mm_batch")
            .shader(SHADER, "main")
            .f64()
            .storage_rw(0, &out_buf)
            .uniform(1, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<MmParams>(), 48);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("michaelis"));
        assert!(SHADER.contains("wang_hash"));
        assert!(SHADER.contains("Params"));
    }
}
