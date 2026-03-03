// SPDX-License-Identifier: AGPL-3.0-or-later
//! BETA F64 - Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) - f64 precision
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//!
//! Applications:
//! - Beta distributions
//! - Bayesian statistics
//! - Binomial coefficients
//! - ML/statistics

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// f64 Beta function evaluator
///
/// Computes B(a,b) = Γ(a)Γ(b)/Γ(a+b) using log-gamma for stability.
pub struct BetaF64 {
    device: Arc<WgpuDevice>,
}

impl BetaF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/beta_f64.wgsl")
    }

    /// Create new Beta f64 operation
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute B(a,b) for each pair
    ///
    /// # Arguments
    /// * `pairs` - Input pairs as interleaved [a₀, b₀, a₁, b₁, ...]
    ///
    /// # Returns
    /// Vector of B(aᵢ, bᵢ) values with f64 precision
    pub fn beta(&self, pairs: &[f64]) -> Result<Vec<f64>> {
        if pairs.is_empty() || !pairs.len().is_multiple_of(2) {
            return Ok(vec![]);
        }

        let num_pairs = pairs.len() / 2;
        let params = Params {
            size: num_pairs as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BetaF64 Input"),
                contents: bytemuck::cast_slice(pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = num_pairs * std::mem::size_of::<f64>();
        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BetaF64 Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BetaF64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "BetaF64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(num_pairs as u32)
            .submit();

        let result: Vec<f64> = self.device.read_buffer_f64(&output_buf, num_pairs)?;
        Ok(result)
    }

    #[cfg(test)]
    fn beta_cpu(&self, pairs: &[f64]) -> Vec<f64> {
        pairs
            .chunks(2)
            .map(|chunk| Self::beta_scalar(chunk[0], chunk[1]))
            .collect()
    }

    #[cfg(test)]
    fn beta_scalar(a: f64, b: f64) -> f64 {
        if a <= 0.0 || b <= 0.0 {
            return f64::NAN;
        }
        // B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
        use std::f64::consts::PI;

        fn lgamma(x: f64) -> f64 {
            if x <= 0.0 {
                return f64::NAN;
            }
            if x < 0.5 {
                return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
            }
            // Lanczos approximation
            let g = 7.0;
            let x_shifted = x - 1.0;
            let mut sum = 0.999_999_999_999_809_9;
            let coeffs = [
                676.5203681218851,
                -1259.1392167224028,
                771.323_428_777_653_1,
                -176.615_029_162_140_6,
                12.507343278686905,
                -0.13857109526572012,
                9.984_369_578_019_572e-6,
                1.5056327351493116e-7,
            ];
            for (i, &c) in coeffs.iter().enumerate() {
                sum += c / (x_shifted + (i + 1) as f64);
            }
            let t = x_shifted + g + 0.5;
            let sqrt_2pi: f64 = 2.5066282746310005;
            sqrt_2pi.ln() + sum.ln() + (x_shifted + 0.5) * t.ln() - t
        }

        let log_beta = lgamma(a) + lgamma(b) - lgamma(a + b);
        log_beta.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_beta_symmetric() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let beta = BetaF64::new(device).unwrap();

        // B(a,b) = B(b,a)
        let pairs = vec![2.0, 3.0, 3.0, 2.0];
        let result = beta.beta(&pairs).unwrap();

        assert!(
            (result[0] - result[1]).abs() < 1e-10,
            "B(2,3) = {} should equal B(3,2) = {}",
            result[0],
            result[1]
        );
    }

    #[tokio::test]
    async fn test_beta_known_values() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let beta = BetaF64::new(device).unwrap();

        // B(1,1) = 1
        let pairs = vec![1.0, 1.0];
        let result = beta.beta(&pairs).unwrap();
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "B(1,1) = {}, expected 1.0",
            result[0]
        );

        // B(2,2) = 1/6
        let pairs = vec![2.0, 2.0];
        let result = beta.beta(&pairs).unwrap();
        let expected = 1.0 / 6.0;
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "B(2,2) = {}, expected {}",
            result[0],
            expected
        );

        // B(3,3) = 1/30
        let pairs = vec![3.0, 3.0];
        let result = beta.beta(&pairs).unwrap();
        let expected = 1.0 / 30.0;
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "B(3,3) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_beta_relation_to_gamma() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let beta = BetaF64::new(device).unwrap();

        // B(n, 1) = 1/n for positive integer n
        for n in 1..=5 {
            let pairs = vec![n as f64, 1.0];
            let result = beta.beta(&pairs).unwrap();
            let expected = 1.0 / n as f64;
            assert!(
                (result[0] - expected).abs() < 1e-6,
                "B({}, 1) = {}, expected {}",
                n,
                result[0],
                expected
            );
        }
    }
}
