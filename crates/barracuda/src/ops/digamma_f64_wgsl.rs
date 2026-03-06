// SPDX-License-Identifier: AGPL-3.0-or-later
//! Digamma ψ(x) (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! ψ(x) = Γ'(x)/Γ(x), the logarithmic derivative of the Gamma function.
//! Applications: Fisher information, Bayesian statistics, neural network regularization

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/digamma_f64.wgsl");
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select shader based on FP64 strategy: native f64 or DF64 auto-rewrite.
fn shader_for_device(device: &WgpuDevice) -> &'static str {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Native | Fp64Strategy::Concurrent => SHADER,
        Fp64Strategy::Hybrid => {
            static DF64_SOURCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                match crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER) {
                    Ok(src) => format!("enable f64;\n{DF64_CORE}\n{src}"),
                    Err(_) => SHADER.to_string(),
                }
            });
            &DF64_SOURCE
        }
    }
}

/// f64 Digamma function evaluator — pipeline-cached
///
/// Computes ψ(x) = d/dx ln(Γ(x)) using reflection + recurrence + asymptotic expansion.
pub struct DigammaF64 {
    device: Arc<WgpuDevice>,
}

impl DigammaF64 {
    /// Create new Digamma f64 operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute ψ(x) for each element
    /// # Arguments
    /// * `x` - Input values
    /// # Returns
    /// Vector of ψ(x) values with f64 precision
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn digamma(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_elementwise(x)
    }

    fn dispatch_elementwise(&self, x: &[f64]) -> Result<Vec<f64>> {
        let n = x.len();
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Digamma Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(n);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self.device.create_uniform_buffer("Digamma Params", &params);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("Digamma BG"),
        );

        let shader_src = shader_for_device(&self.device);
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            "main",
            Some("Digamma Pipeline"),
        );

        let workgroups = n.div_ceil(256) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Digamma Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, n)
    }

    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn digamma_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::digamma_scalar(xi)).collect()
    }

    #[cfg(test)]
    fn digamma_scalar(x: f64) -> f64 {
        use std::f64::consts::PI;

        if x <= 0.0 && x == x.floor() {
            return f64::NAN;
        }

        let mut y = x;
        let mut result = 0.0;

        if y < 0.0 {
            let cot_pi_y = (PI * y).cos() / (PI * y).sin();
            result -= PI * cot_pi_y;
            y = 1.0 - y;
        }

        while y < 6.0 {
            result -= 1.0 / y;
            y += 1.0;
        }

        result + Self::digamma_asymptotic(y)
    }

    #[cfg(test)]
    fn digamma_asymptotic(x: f64) -> f64 {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;

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

        let euler_mascheroni = 0.5772156649015329;
        let result = digamma.digamma(&[2.0]).unwrap();
        let expected = 1.0 - euler_mascheroni;

        assert!(
            (result[0] - expected).abs() < 1e-6,
            "ψ(2) = {}, expected {}",
            result[0],
            expected
        );

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

        let x = 100.0;
        let result = digamma.digamma(&[x]).unwrap();
        let approx = x.ln() - 0.5 / x;

        assert!(
            (result[0] - approx).abs() < 1e-4,
            "ψ({}) = {}, asymptotic approx = {}",
            x,
            result[0],
            approx
        );
    }
}
