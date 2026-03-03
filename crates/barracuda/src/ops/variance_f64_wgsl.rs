// SPDX-License-Identifier: AGPL-3.0-or-later
//! VARIANCE F64 - Variance and standard deviation - f64 precision
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//!
//! Applications:
//! - Statistics
//! - Normalization
//! - Feature scaling

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Simple variance reduction variant (scalar path).
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
struct Params {
    size: u32,
    num_vectors: u32,
    stride: u32,
    ddof: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// f64 Variance/StdDev evaluator
pub struct VarianceF64 {
    device: Arc<WgpuDevice>,
}

impl VarianceF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/variance_f64.wgsl")
    }

    /// Create new Variance f64 operation
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute variance of a vector (population variance, ddof=0)
    pub fn variance(&self, data: &[f64]) -> Result<f64> {
        self.variance_ddof(data, 0)
    }

    /// Compute sample variance (ddof=1)
    pub fn sample_variance(&self, data: &[f64]) -> Result<f64> {
        self.variance_ddof(data, 1)
    }

    /// Compute variance with specified degrees of freedom adjustment
    pub fn variance_ddof(&self, data: &[f64], ddof: usize) -> Result<f64> {
        if data.is_empty() || data.len() <= ddof {
            return Ok(0.0);
        }

        let n = data.len();
        let params = Params {
            size: n as u32,
            num_vectors: 1,
            stride: n as u32,
            ddof: ddof as u32,
            mode: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VarianceF64 Input"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = std::mem::size_of::<f64>();
        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarianceF64 Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VarianceF64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "VarianceF64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &params_buf)
            .dispatch(1, 1, 1)
            .submit();

        let result: Vec<f64> = self.device.read_buffer_f64(&output_buf, 1)?;
        Ok(result[0])
    }

    /// Compute standard deviation (population, ddof=0)
    pub fn std_dev(&self, data: &[f64]) -> Result<f64> {
        Ok(self.variance(data)?.sqrt())
    }

    /// Compute sample standard deviation (ddof=1)
    pub fn sample_std_dev(&self, data: &[f64]) -> Result<f64> {
        Ok(self.sample_variance(data)?.sqrt())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn variance_cpu(data: &[f64], ddof: usize) -> f64 {
        let n = data.len();
        if n <= ddof {
            return 0.0;
        }

        // Two-pass for numerical stability
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

        // Variance of [1, 2, 3, 4, 5] with mean=3
        // Σ(x-μ)² = 4+1+0+1+4 = 10
        // Population variance = 10/5 = 2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = var.variance(&data).unwrap();

        assert!(
            (result - 2.0).abs() < 1e-10,
            "Variance = {}, expected 2.0",
            result
        );
    }

    #[tokio::test]
    async fn test_sample_variance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();

        // Sample variance = 10/4 = 2.5
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = var.sample_variance(&data).unwrap();

        assert!(
            (result - 2.5).abs() < 1e-10,
            "Sample variance = {}, expected 2.5",
            result
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
            "Std dev = {}, expected {}",
            result,
            expected
        );
    }

    #[tokio::test]
    async fn test_variance_constant() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let var = VarianceF64::new(device).unwrap();

        // Variance of constant array is 0
        let data = vec![5.0; 100];
        let result = var.variance(&data).unwrap();

        assert!(
            result.abs() < 1e-10,
            "Variance of constant = {}, expected 0.0",
            result
        );
    }
}
