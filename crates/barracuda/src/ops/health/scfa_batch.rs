// SPDX-License-Identifier: AGPL-3.0-only

//! GPU batch SCFA production — Michaelis-Menten kinetics per metabolite.
//!
//! Each thread computes acetate, propionate, butyrate for one fiber input.
//! Absorbed from healthSpring V19 (Exp079, Exp085).

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/health/scfa_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScfaGpuParams {
    vmax_acetate: f64,
    km_acetate: f64,
    vmax_propionate: f64,
    km_propionate: f64,
    vmax_butyrate: f64,
    km_butyrate: f64,
    n_elements: u32,
    _pad: u32,
}

/// GPU-accelerated batch SCFA production.
pub struct ScfaBatchGpu {
    device: Arc<WgpuDevice>,
}

impl ScfaBatchGpu {
    /// Create a new `ScfaBatchGpu` for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute SCFA production for a batch of fiber concentrations.
    ///
    /// Returns flat `[acetate_0, propionate_0, butyrate_0, acetate_1, ...]`.
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline creation, dispatch, or readback failure.
    pub fn compute(
        &self,
        fiber_values: &[f64],
        scfa_params: &crate::health::microbiome::ScfaParams,
    ) -> Result<Vec<f64>> {
        let n = fiber_values.len();
        let params = ScfaGpuParams {
            vmax_acetate: scfa_params.vmax_acetate,
            km_acetate: scfa_params.km_acetate,
            vmax_propionate: scfa_params.vmax_propionate,
            km_propionate: scfa_params.km_propionate,
            vmax_butyrate: scfa_params.vmax_butyrate,
            km_butyrate: scfa_params.km_butyrate,
            n_elements: n as u32,
            _pad: 0,
        };

        let in_buf = self
            .device
            .create_buffer_f64_init("scfa:input", fiber_values);
        let out_buf = self.device.create_buffer_f64(n * 3)?;
        let params_buf = self.device.create_uniform_buffer("scfa:params", &params);

        let wg_count = (n as u32).div_ceil(256);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "scfa_batch")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &in_buf)
            .storage_rw(1, &out_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n * 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<ScfaGpuParams>(), 56);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("michaelis_menten"));
        assert!(SHADER.contains("acetate"));
        assert!(SHADER.contains("Params"));
    }
}
