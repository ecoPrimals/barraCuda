// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused mean+variance (f64) — single-pass Welford, GPU-resident, pipeline-cached
//!
//! Evolved from separate mean+deviation passes to a single fused dispatch.
//! Absorbed from Kokkos `parallel_reduce` patterns: one kernel, zero intermediate
//! CPU round-trips for chained statistics.

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Fused mean+variance f64 shader (Welford single-pass).
const SHADER_FUSED: &str = include_str!("../shaders/reduce/mean_variance_f64.wgsl");
/// DF64 variant — same Welford algorithm, DF64 core-streaming arithmetic.
const SHADER_FUSED_DF64: &str = include_str!("../shaders/reduce/mean_variance_df64.wgsl");
/// DF64 core arithmetic library (f32-pair).
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select the fused shader based on the device's FP64 strategy.
///
/// Native/Concurrent: use the native f64 Welford shader.
/// Hybrid: use the DF64 variant (polynomial accumulation on f32 cores).
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

/// Simple variance reduction variant (scalar path).
#[must_use]
pub fn wgsl_variance_simple() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
            "../shaders/misc/variance_simple_f64.wgsl"
        ))
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// Special variance shader.
pub const WGSL_VARIANCE_SPECIAL: &str = include_str!("../shaders/special/variance.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FusedParams {
    n: u32,
    ddof: u32,
    _pad0: u32,
    _pad1: u32,
}

/// f64 Variance/StdDev evaluator — fused single-pass Welford
pub struct VarianceF64 {
    device: Arc<WgpuDevice>,
}

impl VarianceF64 {
    /// Create new Variance f64 operation.
    /// # Errors
    /// Never returns an error; always returns `Ok`.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute variance of a vector (population variance, ddof=0).
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn variance(&self, data: &[f64]) -> Result<f64> {
        self.variance_ddof(data, 0)
    }

    /// Compute sample variance (ddof=1).
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn sample_variance(&self, data: &[f64]) -> Result<f64> {
        self.variance_ddof(data, 1)
    }

    /// Compute variance with specified degrees of freedom adjustment.
    /// Uses a fused mean+variance Welford shader — single GPU dispatch,
    /// no intermediate readback between mean and deviation passes.
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn variance_ddof(&self, data: &[f64], ddof: usize) -> Result<f64> {
        if data.is_empty() || data.len() <= ddof {
            return Ok(0.0);
        }

        let result = self.fused_mean_variance(data, ddof)?;
        Ok(result[1])
    }

    /// Compute mean and variance together in a single GPU pass.
    /// Returns `[mean, variance]`.
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn mean_variance(&self, data: &[f64], ddof: usize) -> Result<[f64; 2]> {
        if data.is_empty() || data.len() <= ddof {
            return Ok([0.0, 0.0]);
        }

        let result = self.fused_mean_variance(data, ddof)?;
        Ok([result[0], result[1]])
    }

    /// GPU-resident fused mean+variance — single Welford pass.
    fn fused_mean_variance(&self, data: &[f64], ddof: usize) -> Result<Vec<f64>> {
        let n = data.len();
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FusedMeanVar Input"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(2);

        let params = FusedParams {
            n: n as u32,
            ddof: ddof as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("FusedMeanVar Params", &params);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("FusedMeanVar BG"),
        );

        let shader_src = fused_shader_for_device(&self.device);
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            "main",
            Some("FusedMeanVar Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FusedMeanVar Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, 2)
    }

    /// Compute mean and variance from a GPU-resident buffer — zero host transfer.
    /// The buffer must contain at least `n` f64 values.  Returns `[mean, variance]`.
    /// This is the persistent-buffer path that eliminates the Kokkos dispatch gap.
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn mean_variance_buffer(
        &self,
        buffer: &wgpu::Buffer,
        n: usize,
        ddof: usize,
    ) -> Result<[f64; 2]> {
        if n == 0 || n <= ddof {
            return Ok([0.0, 0.0]);
        }

        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let output_buf = ctx.acquire_pooled_output_f64(2);

        let params = FusedParams {
            n: n as u32,
            ddof: ddof as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("FusedMeanVar:Buf Params", &params);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[buffer, &output_buf, &params_buf],
            Some("FusedMeanVar:Buf BG"),
        );

        let shader_src = fused_shader_for_device(&self.device);
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            "main",
            Some("FusedMeanVar:Buf Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FusedMeanVar:Buf Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        })?;

        let result = self.device.read_buffer_f64(&output_buf, 2)?;
        Ok([result[0], result[1]])
    }

    /// Compute standard deviation (population, ddof=0).
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn std_dev(&self, data: &[f64]) -> Result<f64> {
        Ok(self.variance(data)?.sqrt())
    }

    /// Compute sample standard deviation (ddof=1).
    /// # Errors
    /// Returns [`Err`] if the device is lost, if buffer recording fails, or if
    /// readback fails (e.g. buffer mapping channel closed).
    pub fn sample_std_dev(&self, data: &[f64]) -> Result<f64> {
        Ok(self.sample_variance(data)?.sqrt())
    }

    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn variance_cpu(data: &[f64], ddof: usize) -> f64 {
        let n = data.len();
        if n <= ddof {
            return 0.0;
        }
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let var_sum: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        var_sum / (n - ddof) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_variance_simple() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = var.variance(&data).unwrap();

        assert!(
            (result - 2.0).abs() < 1e-10,
            "Variance = {result}, expected 2.0"
        );
    }

    #[tokio::test]
    async fn test_sample_variance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = var.sample_variance(&data).unwrap();

        assert!(
            (result - 2.5).abs() < 1e-10,
            "Sample variance = {result}, expected 2.5"
        );
    }

    #[tokio::test]
    async fn test_std_dev() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = var.std_dev(&data).unwrap();
        let expected = 2.0_f64.sqrt();

        assert!(
            (result - expected).abs() < 1e-10,
            "Std dev = {result}, expected {expected}"
        );
    }

    #[tokio::test]
    async fn test_variance_constant() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let data = vec![5.0; 100];
        let result = var.variance(&data).unwrap();

        assert!(
            result.abs() < 1e-10,
            "Variance of constant = {result}, expected 0.0"
        );
    }

    #[tokio::test]
    async fn test_fused_mean_variance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let [mean, variance] = var.mean_variance(&data, 0).unwrap();

        assert!((mean - 3.0).abs() < 1e-10, "Mean = {mean}, expected 3.0");
        assert!(
            (variance - 2.0).abs() < 1e-10,
            "Variance = {variance}, expected 2.0"
        );
    }

    #[tokio::test]
    async fn test_fused_mean_variance_large() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();
        let n = 10_000;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let [mean, variance] = var.mean_variance(&data, 0).unwrap();

        let expected_mean = (n - 1) as f64 / 2.0;
        let expected_var = ((n * n - 1) as f64) / 12.0;

        assert!(
            (mean - expected_mean).abs() / expected_mean < 1e-10,
            "Mean = {mean}, expected {expected_mean}"
        );
        assert!(
            (variance - expected_var).abs() / expected_var < 1e-6,
            "Variance = {variance}, expected {expected_var}"
        );
    }
}
