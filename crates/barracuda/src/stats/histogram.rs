// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated histogram via atomic binning.

#[cfg(feature = "gpu")]
use crate::device::compute_pipeline::ComputeDispatch;
#[cfg(feature = "gpu")]
use crate::device::WgpuDevice;
#[cfg(feature = "gpu")]
use crate::error::{BarracudaError, Result};
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HistogramParamsF64 {
    n_values: u32,
    n_bins: u32,
    min_val: f64,
    inv_bin_width: f64,
}

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HistogramParamsF32 {
    n_values: u32,
    n_bins: u32,
    min_val: f32,
    inv_bin_width: f32,
}

/// GPU-accelerated histogram via atomic binning.
#[cfg(feature = "gpu")]
pub struct HistogramGpu {
    device: Arc<WgpuDevice>,
}

#[cfg(feature = "gpu")]
impl HistogramGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch GPU histogram: bin `values` into `n_bins` bins over [min, max].
    pub fn dispatch(&self, values: &[f64], n_bins: u32) -> Result<Vec<u32>> {
        let n_values = values.len();
        if n_values == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "values cannot be empty".to_string(),
            });
        }
        if n_bins == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "n_bins must be > 0".to_string(),
            });
        }

        let (min_val, max_val) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
                (min.min(v), max.max(v))
            });

        let range = max_val - min_val;
        let inv_bin_width = if range > 0.0 && range.is_finite() {
            n_bins as f64 / range
        } else {
            // Constant data or degenerate range: put everything in bin 0
            0.0
        };

        let bins_buf = self.device.create_buffer_u32_zeros(n_bins as usize)?;

        if self.device.has_f64_shaders() {
            self.dispatch_f64(values, n_bins, min_val, inv_bin_width, &bins_buf)
        } else {
            self.dispatch_f32(values, n_bins, min_val, inv_bin_width, &bins_buf)
        }
    }

    fn dispatch_f64(
        &self,
        values: &[f64],
        n_bins: u32,
        min_val: f64,
        inv_bin_width: f64,
        bins_buf: &wgpu::Buffer,
    ) -> Result<Vec<u32>> {
        let n_values = values.len() as u32;

        let params = HistogramParamsF64 {
            n_values,
            n_bins,
            min_val,
            inv_bin_width,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("histogram:params", &params);

        let values_buf = self
            .device
            .create_buffer_f64_init("histogram:values", values);

        let wg_count = n_values.div_ceil(256);
        ComputeDispatch::new(&self.device, "histogram")
            .shader(super::WGSL_HISTOGRAM_F64, "histogram")
            .f64()
            .storage_read(0, &values_buf)
            .storage_rw(1, bins_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        self.device.read_buffer_u32(bins_buf, n_bins as usize)
    }

    fn dispatch_f32(
        &self,
        values: &[f64],
        n_bins: u32,
        min_val: f64,
        inv_bin_width: f64,
        bins_buf: &wgpu::Buffer,
    ) -> Result<Vec<u32>> {
        let n_values = values.len() as u32;

        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        let params = HistogramParamsF32 {
            n_values,
            n_bins,
            min_val: min_val as f32,
            inv_bin_width: inv_bin_width as f32,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("histogram:params", &params);

        let values_buf = self
            .device
            .create_buffer_f32_init("histogram:values", &values_f32);

        let wg_count = n_values.div_ceil(256);
        ComputeDispatch::new(&self.device, "histogram_f32")
            .shader(super::WGSL_HISTOGRAM_F32.as_str(), "histogram")
            .storage_read(0, &values_buf)
            .storage_rw(1, bins_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        self.device.read_buffer_u32(bins_buf, n_bins as usize)
    }
}
