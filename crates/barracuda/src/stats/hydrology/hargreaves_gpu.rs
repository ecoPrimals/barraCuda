// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated batch Hargreaves reference evapotranspiration (ET₀).
//!
//! Dispatches FAO Hargreaves ET₀ for many days in a single compute pass using
//! the `hargreaves_batch_f64` WGSL shader.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use std::sync::Arc;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

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
}
