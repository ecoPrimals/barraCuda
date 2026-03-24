// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hermite Hₙ(x) (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Physicist's Hermite polynomials via three-term recurrence.
//! Applications: quantum harmonic oscillator wavefunctions, nuclear structure,
//! Gaussian quadrature weights, Gaussian-Hermite basis functions

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/hermite_f64.wgsl");
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select shader based on FP64 strategy: native f64 or DF64 auto-rewrite.
///
/// On Hybrid devices the native f64 shader may silently produce zeros,
/// so we require the DF64 rewrite to succeed rather than falling back.
fn shader_for_device(device: &WgpuDevice) -> Result<&'static str> {
    let caps = DeviceCapabilities::from_device(device);
    match caps.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => Ok(SHADER),
        Fp64Strategy::Hybrid => {
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, Arc<str>>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| Arc::from(format!("hermite DF64 rewrite failed: {e}")))
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

/// f64 Hermite polynomial evaluator Hₙ(x) — pipeline-cached
///
/// Computes physicist's Hermite polynomials with full f64 precision
/// using three-term recurrence relation.
pub struct HermiteF64 {
    device: Arc<WgpuDevice>,
}

impl HermiteF64 {
    /// Create new Hermite f64 polynomial operation
    /// # Errors
    /// Returns [`Err`] if device context cannot be obtained.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute Hermite polynomial Hₙ(x) for each element
    /// # Arguments
    /// * `x` - Input values
    /// * `n` - Polynomial order (0, 1, 2, ...)
    /// # Returns
    /// Vector of Hₙ(x) values with f64 precision
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn hermite(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_kernel(x, n, "main")
    }

    /// Compute Hermite function ψₙ(x) (normalized wavefunction)
    /// ψₙ(x) = (2ⁿ·n!·√π)^(-1/2) · Hₙ(x) · exp(-x²/2)
    /// This is the quantum harmonic oscillator eigenfunction.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn hermite_function(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_kernel(x, n, "hermite_function_kernel")
    }

    fn dispatch_kernel(&self, x: &[f64], n: u32, entry_point: &str) -> Result<Vec<f64>> {
        let size = x.len();
        let ctx = get_device_context(&self.device);

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hermite Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(size);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
            _pad0: u32,
            _pad1: u32,
        }

        let params = Params {
            size: size as u32,
            n,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = self.device.create_uniform_buffer("Hermite Params", &params);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("Hermite BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            entry_point,
            Some("Hermite Pipeline"),
        );

        let workgroups = size.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hermite Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, size)
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn hermite_cpu(&self, x: &[f64], n: u32) -> Vec<f64> {
        x.iter().map(|&xi| Self::hermite_scalar(n, xi)).collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn hermite_function_cpu(&self, x: &[f64], n: u32) -> Vec<f64> {
        x.iter()
            .map(|&xi| Self::hermite_function_scalar(n, xi))
            .collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn hermite_scalar(n: u32, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 2.0 * x;
        }

        let mut h_prev = 1.0;
        let mut h_curr = 2.0 * x;

        for k in 1..n {
            let h_next = (2.0 * x).mul_add(h_curr, -(2.0 * (k as f64) * h_prev));
            h_prev = h_curr;
            h_curr = h_next;
        }

        h_curr
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn hermite_function_scalar(n: u32, x: f64) -> f64 {
        let h_n = Self::hermite_scalar(n, x);
        let two_n = 1u64 << n.min(62);
        let n_fact = (1..=n as u64).product::<u64>() as f64;
        let norm = 1.0 / (two_n as f64 * n_fact * std::f64::consts::PI.sqrt()).sqrt();
        norm * h_n * (-x * x / 2.0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_hermite_f64_h0() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = HermiteF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let result = op.hermite(&x, 0).unwrap();

        for &v in &result {
            assert!((v - 1.0).abs() < 1e-10, "H₀ should be 1, got {v}");
        }
    }

    #[test]
    fn test_hermite_f64_h1() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = HermiteF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let result = op.hermite(&x, 1).unwrap();

        for (i, &v) in result.iter().enumerate() {
            let expected = 2.0 * x[i];
            assert!(
                (v - expected).abs() < 1e-10,
                "H₁({}) = {}, expected {}",
                x[i],
                v,
                expected
            );
        }
    }

    #[test]
    fn test_hermite_f64_h2() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = HermiteF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0];
        let result = op.hermite(&x, 2).unwrap();

        let expected = [
            (4.0_f64 * 0.0).mul_add(0.0, -2.0),
            (4.0_f64 * 1.0).mul_add(1.0, -2.0),
            (4.0_f64 * 2.0).mul_add(2.0, -2.0),
        ];

        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-10,
                "H₂({}) = {}, expected {}",
                x[i],
                v,
                expected[i]
            );
        }
    }

    #[test]
    fn test_hermite_function_normalization() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = HermiteF64::new(device).unwrap();

        let x = vec![0.0];
        let psi_0 = op.hermite_function(&x, 0).unwrap();

        let expected = std::f64::consts::PI.powf(-0.25);
        assert!(
            (psi_0[0] - expected).abs() < 1e-6,
            "ψ₀(0) = {}, expected π^(-1/4) = {}",
            psi_0[0],
            expected
        );
    }
}
