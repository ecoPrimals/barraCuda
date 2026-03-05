// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hermite Hₙ(x) (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Physicist's Hermite polynomials via three-term recurrence.
//! Applications: quantum harmonic oscillator wavefunctions, nuclear structure,
//! Gaussian quadrature weights, Gaussian-Hermite basis functions

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/hermite_f64.wgsl");

/// f64 Hermite polynomial evaluator Hₙ(x) — pipeline-cached
///
/// Computes physicist's Hermite polynomials with full f64 precision
/// using three-term recurrence relation.
pub struct HermiteF64 {
    device: Arc<WgpuDevice>,
}

impl HermiteF64 {
    /// Create new Hermite f64 polynomial operation
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute Hermite polynomial Hₙ(x) for each element
    ///
    /// # Arguments
    /// * `x` - Input values
    /// * `n` - Polynomial order (0, 1, 2, ...)
    ///
    /// # Returns
    /// Vector of Hₙ(x) values with f64 precision
    pub fn hermite(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_kernel(x, n, "main")
    }

    /// Compute Hermite function ψₙ(x) (normalized wavefunction)
    ///
    /// ψₙ(x) = (2ⁿ·n!·√π)^(-1/2) · Hₙ(x) · exp(-x²/2)
    ///
    /// This is the quantum harmonic oscillator eigenfunction.
    pub fn hermite_function(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_kernel(x, n, "hermite_function_kernel")
    }

    fn dispatch_kernel(&self, x: &[f64], n: u32, entry_point: &str) -> Result<Vec<f64>> {
        let size = x.len();
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

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

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            SHADER,
            layout_sig,
            entry_point,
            Some("Hermite Pipeline"),
        );

        let workgroups = size.div_ceil(256) as u32;
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

    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn hermite_cpu(&self, x: &[f64], n: u32) -> Vec<f64> {
        x.iter().map(|&xi| Self::hermite_scalar(n, xi)).collect()
    }

    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn hermite_function_cpu(&self, x: &[f64], n: u32) -> Vec<f64> {
        x.iter()
            .map(|&xi| Self::hermite_function_scalar(n, xi))
            .collect()
    }

    #[cfg(test)]
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
            let h_next = 2.0 * x * h_curr - 2.0 * (k as f64) * h_prev;
            h_prev = h_curr;
            h_curr = h_next;
        }

        h_curr
    }

    #[cfg(test)]
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
            assert!((v - 1.0).abs() < 1e-10, "H₀ should be 1, got {}", v);
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
            4.0 * 0.0 * 0.0 - 2.0,
            4.0 * 1.0 * 1.0 - 2.0,
            4.0 * 2.0 * 2.0 - 2.0,
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
