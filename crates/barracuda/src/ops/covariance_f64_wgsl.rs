// SPDX-License-Identifier: AGPL-3.0-or-later
//! Covariance (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Evolved from per-call buffer allocation to pooled `TensorContext` path
//! with cached pipelines and bind groups.
//!
//! Applications: portfolio theory, PCA, Kalman filters

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/covariance_f64.wgsl");
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select shader based on FP64 strategy: native f64 or DF64 auto-rewrite.
///
/// On Hybrid devices the native f64 shader may silently produce zeros,
/// so we require the DF64 rewrite to succeed rather than falling back.
fn shader_for_device(device: &WgpuDevice) -> Result<&'static str> {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => Ok(SHADER),
        Fp64Strategy::Hybrid => {
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, String>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| format!("covariance DF64 rewrite failed: {e}"))
                });
            match DF64_RESULT.as_ref() {
                Ok(src) => Ok(src.as_str()),
                Err(msg) => Err(crate::error::BarracudaError::ShaderCompilation(msg.clone())),
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    num_pairs: u32,
    stride: u32,
    ddof: u32,
}

/// f64 Covariance evaluator — pipeline-cached, buffer-pooled
pub struct CovarianceF64 {
    device: Arc<WgpuDevice>,
}

impl CovarianceF64 {
    /// Create new Covariance f64 operation
    /// # Errors
    /// Returns [`Err`] if the device is invalid or unavailable.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute covariance between two vectors (population covariance, ddof=0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn covariance(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.covariance_ddof(x, y, 0)
    }

    /// Compute sample covariance (ddof=1)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn sample_covariance(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        self.covariance_ddof(x, y, 1)
    }

    /// Compute covariance with specified degrees of freedom adjustment
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn covariance_ddof(&self, x: &[f64], y: &[f64], ddof: usize) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() || x.len() <= ddof {
            return Ok(0.0);
        }

        let n = x.len();
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let params = Params {
            size: n as u32,
            num_pairs: 1,
            stride: n as u32,
            ddof: ddof as u32,
        };

        let x_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CovF64 X"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CovF64 Y"),
                contents: bytemuck::cast_slice(y),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(1);

        let params_buf = self.device.create_uniform_buffer("CovF64 Params", &params);

        let layout_sig = BindGroupLayoutSignature::two_input_reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&x_buf, &y_buf, &output_buf, &params_buf],
            Some("CovF64 BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            "main",
            Some("CovF64 Pipeline"),
        );

        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CovF64 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        })?;

        let result = self.device.read_buffer_f64(&output_buf, 1)?;
        Ok(result[0])
    }

    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn covariance_cpu(x: &[f64], y: &[f64], ddof: usize) -> f64 {
        let n = x.len();
        if n <= ddof {
            return 0.0;
        }

        let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
        let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

        let cov_sum: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        cov_sum / (n - ddof) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_covariance_positive() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let cov = CovarianceF64::new(device).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = cov.covariance(&x, &y).unwrap();

        assert!(result > 0.0, "Covariance should be positive, got {result}");
    }

    #[tokio::test]
    async fn test_covariance_negative() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let cov = CovarianceF64::new(device).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let result = cov.covariance(&x, &y).unwrap();

        assert!(result < 0.0, "Covariance should be negative, got {result}");
    }

    #[tokio::test]
    async fn test_covariance_zero() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let cov = CovarianceF64::new(device).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = cov.covariance(&x, &y).unwrap();

        assert!(
            result.abs() < 1e-10,
            "Covariance with constant should be 0, got {result}"
        );
    }

    #[tokio::test]
    async fn test_covariance_self_is_variance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let cov = CovarianceF64::new(device).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cov_xx = cov.covariance(&x, &x).unwrap();

        assert!(
            (cov_xx - 2.0).abs() < 1e-10,
            "Cov(X,X) = {cov_xx}, expected variance = 2.0"
        );
    }

    #[tokio::test]
    async fn test_sample_covariance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let cov = CovarianceF64::new(device).unwrap();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cov.sample_covariance(&x, &y).unwrap();

        assert!(
            (result - 2.5).abs() < 1e-10,
            "Sample Cov(X,X) = {result}, expected 2.5"
        );
    }
}
