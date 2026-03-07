// SPDX-License-Identifier: AGPL-3.0-or-later
//! LAGUERRE F64 - Generalized Laguerre polynomials - f64 precision WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Applications:
//! - Hydrogen/helium radial wavefunctions (hotSpring)
//! - Nuclear structure calculations
//! - 2D/3D harmonic oscillator basis
//! - Molecular dynamics radial basis

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 Generalized Laguerre polynomial evaluator `L_n^(α)(x)`
///
/// Computes generalized Laguerre polynomials with full f64 precision
/// using three-term recurrence relation.
pub struct LaguerreF64 {
    device: Arc<WgpuDevice>,
}

impl LaguerreF64 {
    /// Create new Laguerre f64 polynomial operation
    /// # Errors
    /// Returns [`Err`] if device context cannot be obtained.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/laguerre_f64.wgsl")
    }

    /// Compute generalized Laguerre polynomial `L_n^(α)(x)` for each element
    /// # Arguments
    /// * `x` - Input values
    /// * `n` - Polynomial degree (0, 1, 2, ...)
    /// * `alpha` - Generalization parameter (0.0 for simple Laguerre)
    /// # Returns
    /// Vector of `L_n^(α)(x)` values with f64 precision
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn laguerre(&self, x: &[f64], n: u32, alpha: f64) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }

        self.laguerre_gpu(x, n, alpha)
    }

    /// Compute simple Laguerre polynomial Lₙ(x) (α = 0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn laguerre_simple(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        self.laguerre(x, n, 0.0)
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn laguerre_cpu(&self, x: &[f64], n: u32, alpha: f64) -> Vec<f64> {
        x.iter()
            .map(|&xi| Self::laguerre_scalar(n, alpha, xi))
            .collect()
    }

    #[allow(dead_code, reason = "CPU scalar helper for GPU parity validation")]
    fn laguerre_scalar(n: u32, alpha: f64, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 1.0 + alpha - x;
        }

        let mut l_prev = 1.0;
        let mut l_curr = 1.0 + alpha - x;

        for k in 1..n {
            let kf = k as f64;
            // Three-term recurrence: n·Lₙ = (2n-1+α-x)·L_{n-1} - (n-1+α)·L_{n-2}
            let l_next =
                ((2.0 * kf + 1.0 + alpha - x) * l_curr - (kf + alpha) * l_prev) / (kf + 1.0);
            l_prev = l_curr;
            l_curr = l_next;
        }

        l_curr
    }

    fn laguerre_gpu(&self, x: &[f64], n: u32, alpha: f64) -> Result<Vec<f64>> {
        let size = x.len();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Laguerre f64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Laguerre f64 Output"),
            size: std::mem::size_of_val(x) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params struct must match WGSL: size, n, _pad0, _pad1, alpha (f64)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
            _pad0: u32,
            _pad1: u32,
            alpha: f64,
        }

        let params = Params {
            size: size as u32,
            n,
            _pad0: 0,
            _pad1: 0,
            alpha,
        };

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Laguerre f64 Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "Laguerre f64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(size as u32)
            .submit()?;

        let result: Vec<f64> = self.device.read_buffer_f64(&output_buf, size)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_laguerre_f64_l0() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = LaguerreF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let result = op.laguerre(&x, 0, 0.0).unwrap();

        // L₀(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-10, "L₀ should be 1, got {v}");
        }
    }

    #[test]
    fn test_laguerre_f64_l1() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = LaguerreF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0, 0.5];
        let result = op.laguerre(&x, 1, 0.0).unwrap();

        // L₁(x) = 1 - x
        for (i, &v) in result.iter().enumerate() {
            let expected = 1.0 - x[i];
            assert!(
                (v - expected).abs() < 1e-10,
                "L₁({}) = {}, expected {}",
                x[i],
                v,
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_f64_l2() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = LaguerreF64::new(device).unwrap();

        let x = vec![0.0, 1.0, 2.0, 0.5];
        let result = op.laguerre(&x, 2, 0.0).unwrap();

        // L₂(x) = (x² - 4x + 2) / 2 = 0.5x² - 2x + 1
        for (i, &v) in result.iter().enumerate() {
            let xi = x[i];
            let expected = 0.5 * xi * xi - 2.0 * xi + 1.0;
            assert!(
                (v - expected).abs() < 1e-10,
                "L₂({xi}) = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_laguerre_f64_generalized() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = LaguerreF64::new(device).unwrap();

        // L₁^(1)(x) = 2 - x (α = 1)
        let x = vec![0.0, 1.0, 2.0];
        let result = op.laguerre(&x, 1, 1.0).unwrap();

        for (i, &v) in result.iter().enumerate() {
            let expected = 2.0 - x[i];
            assert!(
                (v - expected).abs() < 1e-10,
                "L₁^(1)({}) = {}, expected {}",
                x[i],
                v,
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_f64_at_zero() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = LaguerreF64::new(device).unwrap();

        // L_n^(α)(0) = C(n+α, n) = (n+α)! / (n! α!)
        // For α=0: L_n(0) = 1 for all n
        let x = vec![0.0];

        for n in 0..=5 {
            let result = op.laguerre(&x, n, 0.0).unwrap();
            assert!(
                (result[0] - 1.0).abs() < 1e-10,
                "L_{}(0) = {}, expected 1.0",
                n,
                result[0]
            );
        }
    }
}
