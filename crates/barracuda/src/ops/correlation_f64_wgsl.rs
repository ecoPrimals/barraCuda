// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused Pearson correlation (f64) — 5-accumulator single-pass, GPU-resident
//!
//! Evolved from multi-dispatch mean→deviation→covariance to a single fused
//! kernel with 5 accumulators (`sum_x`, `sum_y`, `sum_xx`, `sum_yy`, `sum_xy`).
//! Absorbed from Kokkos `parallel_reduce` with `JoinOp` patterns.

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Fused 5-accumulator correlation shader.
const SHADER_FUSED: &str = include_str!("../shaders/stats/correlation_full_f64.wgsl");
/// DF64 variant — same 5-accumulator algorithm, DF64 core-streaming arithmetic.
const SHADER_FUSED_DF64: &str = include_str!("../shaders/stats/correlation_full_df64.wgsl");
/// DF64 core arithmetic library (f32-pair).
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select the fused correlation shader based on the device's FP64 strategy.
fn fused_shader_for_device(device: &WgpuDevice) -> &'static str {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => SHADER_FUSED,
        Fp64Strategy::Hybrid => {
            static DF64_COMBINED: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                format!("enable f64;\n{DF64_CORE}\n{SHADER_FUSED_DF64}")
            });
            &DF64_COMBINED
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Full correlation result from a single fused dispatch.
#[derive(Debug, Clone, Copy)]
pub struct CorrelationResult {
    /// Mean of the x vector.
    pub mean_x: f64,
    /// Mean of the y vector.
    pub mean_y: f64,
    /// Population variance of x.
    pub var_x: f64,
    /// Population variance of y.
    pub var_y: f64,
    /// Pearson correlation coefficient.
    pub pearson_r: f64,
}

impl CorrelationResult {
    /// Population covariance derived from the fused accumulators.
    ///
    /// `cov(x,y) = r * sqrt(var_x) * sqrt(var_y)`
    #[must_use]
    pub fn covariance(&self) -> f64 {
        self.pearson_r * self.var_x.sqrt() * self.var_y.sqrt()
    }

    /// R-squared (coefficient of determination) between observed (x) and
    /// simulated (y).
    ///
    /// Equivalent to `1 - SS_res / SS_tot` when x is observed and y is
    /// simulated, but computed from the fused accumulators without a
    /// second pass.
    ///
    /// For the common case where R² = r², this returns `pearson_r.powi(2)`.
    #[must_use]
    pub fn r_squared(&self) -> f64 {
        self.pearson_r * self.pearson_r
    }
}

/// f64 Pearson correlation evaluator — fused single-pass.
pub struct CorrelationF64 {
    device: Arc<WgpuDevice>,
}

impl CorrelationF64 {
    /// Create new Correlation f64 operation.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute Pearson correlation coefficient between two vectors.
    /// r = cov(x,y) / (std(x) * std(y))
    /// Single GPU dispatch — no intermediate readbacks.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn correlation(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        Ok(self.correlation_full(x, y)?.pearson_r)
    }

    /// Coefficient of determination (R²) between observed and simulated.
    ///
    /// Single GPU dispatch, zero CPU round-trips. Returns `pearson_r²` which
    /// equals the true R² for simple linear models.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn r_squared(&self, observed: &[f64], simulated: &[f64]) -> Result<f64> {
        Ok(self.correlation_full(observed, simulated)?.r_squared())
    }

    /// Population covariance between two vectors. Single GPU dispatch.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn covariance(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        Ok(self.correlation_full(x, y)?.covariance())
    }

    /// Compute full correlation statistics in a single fused dispatch.
    /// Returns means, variances, and Pearson r — all from one kernel launch.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
    /// readback fails.
    pub fn correlation_full(&self, x: &[f64], y: &[f64]) -> Result<CorrelationResult> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(CorrelationResult {
                mean_x: 0.0,
                mean_y: 0.0,
                var_x: 0.0,
                var_y: 0.0,
                pearson_r: 0.0,
            });
        }

        let n = x.len();
        let ctx = get_device_context(&self.device);

        let x_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CorrFused X"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CorrFused Y"),
                contents: bytemuck::cast_slice(y),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(5);

        let params = Params {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("CorrFused Params", &params);

        let layout_sig = BindGroupLayoutSignature {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 1,
        };

        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&x_buf, &y_buf, &output_buf, &params_buf],
            Some("CorrFused BG"),
        );

        let shader_src = fused_shader_for_device(&self.device);
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("CorrFused Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CorrFused Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        })?;

        let result = self.device.read_buffer_f64(&output_buf, 5)?;

        Ok(CorrelationResult {
            mean_x: result[0],
            mean_y: result[1],
            var_x: result[2],
            var_y: result[3],
            pearson_r: result[4],
        })
    }

    /// Compute full correlation from GPU-resident buffers — zero host transfer.
    /// Both buffers must contain at least `n` f64 values.
    /// This is the persistent-buffer path that eliminates the Kokkos dispatch gap.
    /// # Errors
    /// Returns [`Err`] if device context recording fails, the device is lost, or
    /// buffer readback fails.
    pub fn correlation_full_buffer(
        &self,
        x_buffer: &wgpu::Buffer,
        y_buffer: &wgpu::Buffer,
        n: usize,
    ) -> Result<CorrelationResult> {
        if n == 0 {
            return Ok(CorrelationResult {
                mean_x: 0.0,
                mean_y: 0.0,
                var_x: 0.0,
                var_y: 0.0,
                pearson_r: 0.0,
            });
        }

        let ctx = get_device_context(&self.device);

        let output_buf = ctx.acquire_pooled_output_f64(5);

        let params = Params {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("CorrFused:Buf Params", &params);

        let layout_sig = BindGroupLayoutSignature {
            read_only_buffers: 2,
            read_write_buffers: 1,
            uniform_buffers: 1,
        };

        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[x_buffer, y_buffer, &output_buf, &params_buf],
            Some("CorrFused:Buf BG"),
        );

        let shader_src = fused_shader_for_device(&self.device);
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("CorrFused:Buf Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CorrFused:Buf Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        })?;

        let result = self.device.read_buffer_f64(&output_buf, 5)?;

        Ok(CorrelationResult {
            mean_x: result[0],
            mean_y: result[1],
            var_x: result[2],
            var_y: result[3],
            pearson_r: result[4],
        })
    }

    /// Compute Pearson r from GPU-resident buffers — zero host transfer.
    /// # Errors
    /// Returns [`Err`] if device context recording fails, the device is lost, or
    /// buffer readback fails.
    pub fn correlation_buffer(
        &self,
        x_buffer: &wgpu::Buffer,
        y_buffer: &wgpu::Buffer,
        n: usize,
    ) -> Result<f64> {
        Ok(self
            .correlation_full_buffer(x_buffer, y_buffer, n)?
            .pearson_r)
    }

    #[expect(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn correlation_cpu(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        if n == 0 {
            return 0.0;
        }
        let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        if denom < 1e-15 {
            return 0.0;
        }
        cov / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_correlation_perfect_positive() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = corr.correlation(&x, &y).unwrap();

        assert!(
            (result - 1.0).abs() < 1e-10,
            "Correlation should be 1.0, got {result}"
        );
    }

    #[tokio::test]
    async fn test_correlation_perfect_negative() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let result = corr.correlation(&x, &y).unwrap();

        assert!(
            (result + 1.0).abs() < 1e-10,
            "Correlation should be -1.0, got {result}"
        );
    }

    #[tokio::test]
    async fn test_correlation_self() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![1.0, 3.0, 7.0, 2.5, 9.0];
        let result = corr.correlation(&x, &x).unwrap();

        assert!(
            (result - 1.0).abs() < 1e-10,
            "Self-correlation should be 1.0, got {result}"
        );
    }

    #[tokio::test]
    async fn test_correlation_uncorrelated() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![1.0, 0.0, -1.0, 0.0];
        let y = vec![0.0, 1.0, 0.0, -1.0];
        let result = corr.correlation(&x, &y).unwrap();

        assert!(
            result.abs() < 1e-10,
            "Orthogonal vectors should have correlation ~0, got {result}"
        );
    }

    #[tokio::test]
    async fn test_correlation_bounds() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![2.3, 5.1, 1.2, 8.7, 3.3, 6.8, 4.2];
        let y = vec![1.5, 4.2, 2.8, 7.1, 5.5, 3.9, 6.1];
        let result = corr.correlation(&x, &y).unwrap();

        assert!(
            (-1.0..=1.0).contains(&result),
            "Correlation must be in [-1, 1], got {result}"
        );
    }

    #[tokio::test]
    async fn test_correlation_full_returns_all_stats() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let corr = CorrelationF64::new(device).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = corr.correlation_full(&x, &y).unwrap();

        assert!(
            (result.mean_x - 3.0).abs() < 1e-10,
            "mean_x = {}",
            result.mean_x
        );
        assert!(
            (result.mean_y - 6.0).abs() < 1e-10,
            "mean_y = {}",
            result.mean_y
        );
        assert!(
            (result.var_x - 2.0).abs() < 1e-10,
            "var_x = {}",
            result.var_x
        );
        assert!(
            (result.var_y - 8.0).abs() < 1e-10,
            "var_y = {}",
            result.var_y
        );
        assert!(
            (result.pearson_r - 1.0).abs() < 1e-10,
            "r = {}",
            result.pearson_r
        );
    }
}
