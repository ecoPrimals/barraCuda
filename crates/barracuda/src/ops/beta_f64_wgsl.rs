// SPDX-License-Identifier: AGPL-3.0-or-later
//! Beta B(a,b) (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! B(a,b) = Γ(a)Γ(b)/Γ(a+b), computed via log-gamma for stability.
//! Applications: Beta distributions, Bayesian statistics, binomial coefficients

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/beta_f64.wgsl");
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
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, Arc<str>>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| Arc::from(format!("beta DF64 rewrite failed: {e}")))
                });
            match DF64_RESULT.as_ref() {
                Ok(src) => Ok(src),
                Err(msg) => Err(crate::error::BarracudaError::ShaderCompilation(Arc::clone(
                    msg,
                ))),
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// f64 Beta function evaluator — pipeline-cached
///
/// Computes B(a,b) = Γ(a)Γ(b)/Γ(a+b) using log-gamma for stability.
pub struct BetaF64 {
    device: Arc<WgpuDevice>,
}

impl BetaF64 {
    /// Create new Beta f64 operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
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
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn beta(&self, pairs: &[f64]) -> Result<Vec<f64>> {
        if pairs.is_empty() || !pairs.len().is_multiple_of(2) {
            return Ok(vec![]);
        }

        let num_pairs = pairs.len() / 2;
        let ctx = get_device_context(&self.device);

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
                label: Some("Beta Input"),
                contents: bytemuck::cast_slice(pairs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(num_pairs);

        let params_buf = self.device.create_uniform_buffer("Beta Params", &params);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("Beta BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("Beta Pipeline"),
        );

        let workgroups = num_pairs.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Beta Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, num_pairs)
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn beta_cpu(&self, pairs: &[f64]) -> Vec<f64> {
        pairs
            .chunks(2)
            .map(|chunk| Self::beta_scalar(chunk[0], chunk[1]))
            .collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn beta_scalar(a: f64, b: f64) -> f64 {
        if a <= 0.0 || b <= 0.0 {
            return f64::NAN;
        }
        use std::f64::consts::PI;

        fn lgamma(x: f64) -> f64 {
            if x <= 0.0 {
                return f64::NAN;
            }
            if x < 0.5 {
                return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
            }
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

        let pairs = vec![1.0, 1.0];
        let result = beta.beta(&pairs).unwrap();
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "B(1,1) = {}, expected 1.0",
            result[0]
        );

        let pairs = vec![2.0, 2.0];
        let result = beta.beta(&pairs).unwrap();
        let expected = 1.0 / 6.0;
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "B(2,2) = {}, expected {}",
            result[0],
            expected
        );

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
