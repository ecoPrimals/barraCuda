// SPDX-License-Identifier: AGPL-3.0-or-later
//! DIGAMMA F64 - Digamma function ψ(x) = Γ'(x)/Γ(x) - f64 precision
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//!
//! Note: Uses GPU for f64 log/sin/cos when available.
//!
//! Applications:
//! - Fisher information
//! - Bayesian statistics
//! - Neural network regularization

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// f64 Digamma function evaluator
///
/// Computes ψ(x) = d/dx ln(Γ(x)) using reflection + recurrence + asymptotic expansion.
pub struct DigammaF64 {
    device: Arc<WgpuDevice>,
}

impl DigammaF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/digamma_f64.wgsl")
    }

    /// Create new Digamma f64 operation
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute ψ(x) for each element
    ///
    /// # Arguments
    /// * `x` - Input values
    ///
    /// # Returns
    /// Vector of ψ(x) values with f64 precision
    pub fn digamma(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }

        let n = x.len();
        let params = Params {
            size: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DigammaF64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = std::mem::size_of_val(x);
        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DigammaF64 Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DigammaF64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "DigammaF64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(n as u32)
            .submit();

        let result: Vec<f64> = self.device.read_buffer_f64(&output_buf, n)?;
        Ok(result)
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn digamma_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::digamma_scalar(xi)).collect()
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn digamma_scalar(x: f64) -> f64 {
        use std::f64::consts::PI;

        // Non-positive integer: pole
        if x <= 0.0 && x == x.floor() {
            return f64::NAN;
        }

        let mut y = x;
        let mut result = 0.0;

        // Reflection formula for x < 0
        if y < 0.0 {
            let cot_pi_y = (PI * y).cos() / (PI * y).sin();
            result -= PI * cot_pi_y;
            y = 1.0 - y;
        }

        // Recurrence to shift to larger argument
        while y < 6.0 {
            result -= 1.0 / y;
            y += 1.0;
        }

        // Asymptotic expansion for y >= 6
        result + Self::digamma_asymptotic(y)
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn digamma_asymptotic(x: f64) -> f64 {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;

        // Bernoulli number coefficients
        const B2: f64 = 1.0 / 12.0;
        const B4: f64 = -1.0 / 120.0;
        const B6: f64 = 1.0 / 252.0;
        const B8: f64 = -1.0 / 240.0;
        const B10: f64 = 1.0 / 132.0;
        const B12: f64 = -691.0 / 32_760.0;

        let mut sum = x.ln() - 0.5 * inv_x;
        let mut term = inv_x2;

        sum -= B2 * term;
        term *= inv_x2;
        sum -= B4 * term;
        term *= inv_x2;
        sum -= B6 * term;
        term *= inv_x2;
        sum -= B8 * term;
        term *= inv_x2;
        sum -= B10 * term;
        term *= inv_x2;
        sum -= B12 * term;

        sum
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_digamma_at_1() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let digamma = DigammaF64::new(device).unwrap();

        // ψ(1) = -γ (Euler-Mascheroni constant)
        let euler_mascheroni = 0.5772156649015329;
        let result = digamma.digamma(&[1.0]).unwrap();

        assert!(
            (result[0] + euler_mascheroni).abs() < 1e-6,
            "ψ(1) = {}, expected -γ = {}",
            result[0],
            -euler_mascheroni
        );
    }

    #[tokio::test]
    async fn test_digamma_recurrence() {
        let run = |device: std::sync::Arc<crate::device::WgpuDevice>| {
            let digamma = DigammaF64::new(device)?;
            // ψ(x+1) = ψ(x) + 1/x
            for x in [1.0, 2.0, 3.0, 4.5, 7.3] {
                let result = digamma.digamma(&[x, x + 1.0])?;
                let psi_x = result[0];
                let psi_x1 = result[1];

                assert!(
                    (psi_x1 - psi_x - 1.0 / x).abs() < 1e-6,
                    "ψ({}) + 1/{} = {} should equal ψ({}) = {}",
                    x,
                    x,
                    psi_x + 1.0 / x,
                    x + 1.0,
                    psi_x1
                );
            }
            Ok::<(), crate::error::BarracudaError>(())
        };

        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        match run(device) {
            Ok(()) => {}
            Err(e) if e.is_device_lost() => {
                tracing::warn!("device lost in digamma recurrence, retrying");
                let fresh = get_test_device_if_f64_gpu_available()
                    .await
                    .expect("f64 GPU unavailable on retry");
                run(fresh).expect("failed on retry after device recovery");
            }
            Err(e) => panic!("test failed: {e}"),
        }
    }

    #[tokio::test]
    async fn test_digamma_known_values() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let digamma = DigammaF64::new(device).unwrap();

        // ψ(2) = 1 - γ
        let euler_mascheroni = 0.5772156649015329;
        let result = digamma.digamma(&[2.0]).unwrap();
        let expected = 1.0 - euler_mascheroni;

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "ψ(2) = {}, expected {}",
            result[0],
            expected
        );

        // ψ(1/2) = -γ - 2*ln(2)
        let result = digamma.digamma(&[0.5]).unwrap();
        let expected = -euler_mascheroni - 2.0 * 2.0_f64.ln();

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "ψ(0.5) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_digamma_large_x() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let digamma = DigammaF64::new(device).unwrap();

        // For large x, ψ(x) ≈ ln(x) - 1/(2x)
        let x = 100.0;
        let result = digamma.digamma(&[x]).unwrap();
        let approx = x.ln() - 0.5 / x;

        // The actual value is more accurate than the simple approximation
        // Asymptotic expansion includes higher order terms that improve accuracy
        assert!(
            (result[0] - approx).abs() < 1e-4,
            "ψ({}) = {}, asymptotic approx = {}",
            x,
            result[0],
            approx
        );
    }
}
