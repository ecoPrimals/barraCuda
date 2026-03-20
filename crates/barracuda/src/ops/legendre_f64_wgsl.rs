// SPDX-License-Identifier: AGPL-3.0-or-later
//! LEGENDRE F64 - Legendre polynomials and associated Legendre functions - f64 precision WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Applications:
//! - Spherical harmonics expansion
//! - Angular momentum in quantum mechanics
//! - Multipole expansion in electrostatics
//! - Gravitational field modeling

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 Legendre polynomial evaluator Pₙ(x) and Pₙᵐ(x)
///
/// Computes Legendre polynomials and associated Legendre functions
/// with full f64 precision using three-term recurrence relations.
pub struct LegendreF64 {
    device: Arc<WgpuDevice>,
}

impl LegendreF64 {
    /// Create new Legendre f64 polynomial operation
    /// # Errors
    /// Returns [`Err`] if device context cannot be obtained.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/legendre_f64.wgsl")
    }

    /// Compute Legendre polynomial Pₙ(x) for each element
    /// # Arguments
    /// * `x` - Input values (should be in [-1, 1] for standard domain)
    /// * `n` - Polynomial degree (0, 1, 2, ...)
    /// # Returns
    /// Vector of Pₙ(x) values with f64 precision
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn legendre(&self, x: &[f64], n: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }

        self.execute_kernel(x, n, 0, false)
    }

    /// Compute associated Legendre function Pₙᵐ(x) for each element
    /// Uses Condon-Shortley phase convention.
    /// # Arguments
    /// * `x` - Input values (should be in [-1, 1], typically cos(θ))
    /// * `n` - Degree (0, 1, 2, ...)
    /// * `m` - Order (0 ≤ m ≤ n)
    /// # Returns
    /// Vector of Pₙᵐ(x) values with f64 precision
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer readback fails
    /// (e.g. device lost).
    pub fn assoc_legendre(&self, x: &[f64], n: u32, m: u32) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        if m > n {
            return Ok(vec![0.0; x.len()]);
        }

        self.execute_kernel(x, n, m, true)
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn legendre_cpu(&self, x: &[f64], n: u32) -> Vec<f64> {
        x.iter().map(|&xi| Self::legendre_scalar(n, xi)).collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn assoc_legendre_cpu(&self, x: &[f64], n: u32, m: u32) -> Vec<f64> {
        x.iter()
            .map(|&xi| Self::assoc_legendre_scalar(n, m, xi))
            .collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn legendre_scalar(n: u32, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return x;
        }

        let mut p_prev = 1.0;
        let mut p_curr = x;

        for k in 1..n {
            let k_f64 = k as f64;
            let p_next = ((2.0 * k_f64 + 1.0) * x * p_curr - k_f64 * p_prev) / (k_f64 + 1.0);
            p_prev = p_curr;
            p_curr = p_next;
        }

        p_curr
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn double_factorial(m: u32) -> f64 {
        if m == 0 {
            return 1.0;
        }
        let mut r = 1.0;
        for k in 1..=m {
            r *= (2 * k - 1) as f64;
        }
        r
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn assoc_legendre_scalar(n: u32, m: u32, x: f64) -> f64 {
        if m > n {
            return 0.0;
        }
        if m == 0 {
            return Self::legendre_scalar(n, x);
        }

        let sin_sq = 1.0 - x * x;
        if sin_sq <= 0.0 {
            return 0.0;
        }
        let sin_theta_m = sin_sq.sqrt().powf(m as f64);

        // P_m^m with Condon-Shortley phase
        let mut pmm = Self::double_factorial(m) * sin_theta_m;
        if m % 2 == 1 {
            pmm = -pmm;
        }

        if n == m {
            return pmm;
        }

        // P_{m+1}^m
        let pm1m = x * (2 * m + 1) as f64 * pmm;

        if n == m + 1 {
            return pm1m;
        }

        // Upward recurrence
        let mut pl_minus2 = pmm;
        let mut pl_minus1 = pm1m;

        for l in (m + 2)..=n {
            let l_f64 = l as f64;
            let m_f64 = m as f64;
            let pl = ((2.0 * l_f64 - 1.0) * x * pl_minus1 - (l_f64 + m_f64 - 1.0) * pl_minus2)
                / (l_f64 - m_f64);
            pl_minus2 = pl_minus1;
            pl_minus1 = pl;
        }

        pl_minus1
    }

    fn execute_kernel(&self, x: &[f64], n: u32, m: u32, is_assoc: bool) -> Result<Vec<f64>> {
        let size = x.len();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Legendre f64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Legendre f64 Output"),
            size: std::mem::size_of_val(x) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
            m: u32,
            is_assoc: u32,
        }

        let params = Params {
            size: size as u32,
            n,
            m,
            is_assoc: u32::from(is_assoc),
        };

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Legendre f64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "Legendre f64")
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

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_legendre_p0() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = leg.legendre(&x, 0)?;

        // P₀(x) = 1 for all x
        for val in result {
            assert!((val - 1.0).abs() < 1e-10, "P₀(x) should be 1, got {val}");
        }
        Ok(())
    }

    #[test]
    fn test_legendre_p1() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = leg.legendre(&x, 1)?;

        // P₁(x) = x
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - x[i]).abs() < 1e-10,
                "P₁({}) = {}, expected {}",
                x[i],
                val,
                x[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_legendre_p2() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        let x = vec![-1.0, 0.0, 1.0];
        let result = leg.legendre(&x, 2)?;

        // P₂(x) = (3x² - 1)/2
        let expected: Vec<f64> = x.iter().map(|&xi| (3.0 * xi * xi - 1.0) / 2.0).collect();

        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-10,
                "P₂({}) = {}, expected {}",
                x[i],
                val,
                expected[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_assoc_legendre_p11() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        let x = vec![0.0, 0.5, 0.8660254]; // cos(90°), cos(60°), cos(30°)
        let result = leg.assoc_legendre(&x, 1, 1)?;

        // P₁¹(x) = -√(1-x²) [Condon-Shortley]
        for (i, &val) in result.iter().enumerate() {
            let expected = -(1.0 - x[i] * x[i]).sqrt();
            assert!(
                (val - expected).abs() < 1e-10,
                "P₁¹({}) = {}, expected {}",
                x[i],
                val,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_assoc_legendre_boundary() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        // At x = ±1, Pₙᵐ = 0 for m > 0
        let x = vec![-1.0, 1.0];
        let result = leg.assoc_legendre(&x, 3, 2)?;

        for (i, &val) in result.iter().enumerate() {
            assert!(val.abs() < 1e-10, "P₃²({}) should be 0, got {}", x[i], val);
        }
        Ok(())
    }

    #[test]
    fn test_legendre_large() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let leg = LegendreF64::new(device)?;

        // Large input to trigger GPU path
        let x: Vec<f64> = (0..1000).map(|i| -1.0 + 2.0 * (i as f64) / 999.0).collect();
        let result = leg.legendre(&x, 5)?;

        assert_eq!(result.len(), 1000);

        // Check boundary values: P₅(-1) = -1, P₅(1) = 1
        assert!(
            (result[0] + 1.0).abs() < 1e-8,
            "P₅(-1) = {}, expected -1",
            result[0]
        );
        assert!(
            (result[999] - 1.0).abs() < 1e-8,
            "P₅(1) = {}, expected 1",
            result[999]
        );

        Ok(())
    }
}
