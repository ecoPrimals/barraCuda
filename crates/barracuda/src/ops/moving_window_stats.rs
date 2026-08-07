// SPDX-License-Identifier: AGPL-3.0-or-later

//! Moving Window Statistics — GPU-accelerated sliding window mean/var/min/max
//!
//! Computes four summary statistics over a 1D sliding window in a single
//! GPU dispatch, ideal for real-time `IoT` sensor streams.
//!
//! Provenance: airSpring precision agriculture / wetSpring monitoring

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// WGSL shader for moving window mean/variance/min/max (f32, downcast from f64).
pub const WGSL_MOVING_WINDOW_STATS: &str = include_str!("../shaders/stats/moving_window_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MovingWindowParams {
    n: u32,
    window: u32,
    n_out: u32,
    _pad: u32,
}

/// Result of a moving window statistics computation.
#[derive(Debug, Clone)]
pub struct MovingWindowResult {
    /// Mean per window.
    pub mean: Vec<f32>,
    /// Variance per window.
    pub variance: Vec<f32>,
    /// Minimum per window.
    pub min: Vec<f32>,
    /// Maximum per window.
    pub max: Vec<f32>,
}

/// GPU-accelerated moving window statistics (mean, variance, min, max).
pub struct MovingWindowStats {
    device: Arc<WgpuDevice>,
}

impl MovingWindowStats {
    /// Create a moving window stats instance for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute moving window statistics over `input` with given `window_size`.
    ///
    /// Returns vectors of length `input.len() - window_size + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(&self, input: &[f32], window_size: usize) -> Result<MovingWindowResult> {
        let n = input.len();
        if window_size == 0 || window_size > n {
            return Err(BarracudaError::invalid_input(format!(
                "window_size {window_size} must be in [1, {n}]"
            )));
        }

        self.compute_gpu(input, window_size)
    }

    fn compute_gpu(&self, input: &[f32], window_size: usize) -> Result<MovingWindowResult> {
        let n = crate::utils::checked_u32(input.len(), "moving_window input_len")?;
        let n_out =
            crate::utils::checked_u32(input.len() - window_size + 1, "moving_window output_len")?;

        let d = self.device.device();
        let q = self.device.queue();

        let params = MovingWindowParams {
            n,
            window: crate::utils::checked_u32(window_size, "moving_window window_size")?,
            n_out,
            _pad: 0,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mw_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let input_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mw_input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_size = (n_out as usize) * std::mem::size_of::<f32>();
        let make_out_buf = |label: &str| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: out_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let mean_buf = make_out_buf("mw_mean");
        let var_buf = make_out_buf("mw_var");
        let min_buf = make_out_buf("mw_min");
        let max_buf = make_out_buf("mw_max");

        ComputeDispatch::new(&self.device, "moving_window_stats")
            .shader(WGSL_MOVING_WINDOW_STATS, "moving_window_stats")
            .uniform(0, &params_buf)
            .storage_read(1, &input_buf)
            .storage_rw(2, &mean_buf)
            .storage_rw(3, &var_buf)
            .storage_rw(4, &min_buf)
            .storage_rw(5, &max_buf)
            .dispatch_1d(n_out)
            .submit()?;

        let staging_size = out_size as u64;
        let make_staging = |label: &str| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let s_mean = make_staging("s_mean");
        let s_var = make_staging("s_var");
        let s_min = make_staging("s_min");
        let s_max = make_staging("s_max");

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("mw_readback"),
            });
        encoder.copy_buffer_to_buffer(&mean_buf, 0, &s_mean, 0, staging_size);
        encoder.copy_buffer_to_buffer(&var_buf, 0, &s_var, 0, staging_size);
        encoder.copy_buffer_to_buffer(&min_buf, 0, &s_min, 0, staging_size);
        encoder.copy_buffer_to_buffer(&max_buf, 0, &s_max, 0, staging_size);

        q.submit(Some(encoder.finish()));

        let n_out = n_out as usize;
        let mean = self.device.map_staging_buffer::<f32>(&s_mean, n_out)?;
        let variance = self.device.map_staging_buffer::<f32>(&s_var, n_out)?;
        let min_vals = self.device.map_staging_buffer::<f32>(&s_min, n_out)?;
        let max_vals = self.device.map_staging_buffer::<f32>(&s_max, n_out)?;

        Ok(MovingWindowResult {
            mean,
            variance,
            min: min_vals,
            max: max_vals,
        })
    }

    #[cfg(test)]
    fn compute_cpu(input: &[f32], window_size: usize) -> MovingWindowResult {
        let n_out = input.len() - window_size + 1;
        let w = window_size as f32;
        let mut mean = Vec::with_capacity(n_out);
        let mut variance = Vec::with_capacity(n_out);
        let mut min_vals = Vec::with_capacity(n_out);
        let mut max_vals = Vec::with_capacity(n_out);

        for i in 0..n_out {
            let window = &input[i..i + window_size];
            let sum: f32 = window.iter().sum();
            let sum_sq: f32 = window.iter().map(|v| v * v).sum();
            let m = sum / w;
            let v = m.mul_add(-m, sum_sq / w).max(0.0);

            let lo = window.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            mean.push(m);
            variance.push(v);
            min_vals.push(lo);
            max_vals.push(hi);
        }

        MovingWindowResult {
            mean,
            variance,
            min: min_vals,
            max: max_vals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_window_size() {
        let device = crate::device::test_pool::get_test_device_sync();
        let mw = MovingWindowStats::new(device);
        let result = mw.compute(&[1.0, 2.0, 3.0], 0);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_window_larger_than_input() {
        let device = crate::device::test_pool::get_test_device_sync();
        let mw = MovingWindowStats::new(device);
        let result = mw.compute(&[1.0, 2.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_constant_signal() {
        let input = vec![5.0f32; 100];
        let result = MovingWindowStats::compute_cpu(&input, 10);

        assert_eq!(result.mean.len(), 91);
        for &m in &result.mean {
            assert!((m - 5.0).abs() < 1e-5);
        }
        for &v in &result.variance {
            assert!(v.abs() < 1e-5);
        }
        for &lo in &result.min {
            assert!((lo - 5.0).abs() < 1e-5);
        }
        for &hi in &result.max {
            assert!((hi - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn cpu_ramp() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let result = MovingWindowStats::compute_cpu(&input, 5);

        assert_eq!(result.mean.len(), 16);
        assert!((result.mean[0] - 2.0).abs() < 1e-5);
        assert!((result.min[0] - 0.0).abs() < 1e-5);
        assert!((result.max[0] - 4.0).abs() < 1e-5);
        assert!((result.variance[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn cpu_single_element_window() {
        let input: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let result = MovingWindowStats::compute_cpu(&input, 1);
        assert_eq!(result.mean.len(), 5);
        for (i, &m) in result.mean.iter().enumerate() {
            assert!((m - i as f32).abs() < 1e-5);
        }
        for &v in &result.variance {
            assert!(v.abs() < 1e-5, "single-element window has zero variance");
        }
    }

    #[test]
    fn cpu_full_window() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = MovingWindowStats::compute_cpu(&input, 5);
        assert_eq!(result.mean.len(), 1);
        assert!((result.mean[0] - 3.0).abs() < 1e-5);
        assert!((result.min[0] - 1.0).abs() < 1e-5);
        assert!((result.max[0] - 5.0).abs() < 1e-5);
    }
}
