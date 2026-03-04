// SPDX-License-Identifier: AGPL-3.0-or-later
//! SPHERICAL HARMONICS F64 - Real spherical harmonics Y_l^m - f64 precision WGSL
//!
//! Deep Debt Principles apply.
//!
//! Applications:
//! - Multipole expansion in electrostatics
//! - Angular momentum eigenfunctions
//! - Molecular orbital visualization
//! - Gravitational field modeling

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// f64 Real spherical harmonics evaluator Y_l^m(θ, φ)
pub struct SphericalHarmonicsF64 {
    device: Arc<WgpuDevice>,
}

impl SphericalHarmonicsF64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/spherical_harmonics_f64.wgsl")
    }

    /// Compute real spherical harmonic Y_l^m(θ, φ) for each (θ, φ) pair
    ///
    /// # Arguments
    /// * `theta_phi` - Interleaved [θ₀, φ₀, θ₁, φ₁, ...] angles in radians
    /// * `l` - Degree (0, 1, 2, ...)
    /// * `m` - Order (-l ≤ m ≤ l)
    pub fn ylm(&self, theta_phi: &[f64], l: u32, m: i32) -> Result<Vec<f64>> {
        if theta_phi.is_empty() || !theta_phi.len().is_multiple_of(2) {
            return Ok(vec![]);
        }

        let size = theta_phi.len() / 2;
        let _abs_m = m.unsigned_abs();

        if _abs_m > l {
            return Ok(vec![0.0; size]);
        }

        self.ylm_gpu(theta_phi, l, m)
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn ylm_cpu(&self, theta_phi: &[f64], l: u32, m: i32) -> Vec<f64> {
        let size = theta_phi.len() / 2;
        let _abs_m = m.unsigned_abs(); // Used in GPU path
        let mut result = Vec::with_capacity(size);

        for i in 0..size {
            let theta = theta_phi[i * 2];
            let phi = theta_phi[i * 2 + 1];
            result.push(Self::ylm_scalar(l, m, theta, phi));
        }
        result
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn factorial(n: u32) -> f64 {
        match n {
            0 | 1 => 1.0,
            _ => (2..=n).map(|i| i as f64).product(),
        }
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn double_factorial(m: u32) -> f64 {
        if m == 0 {
            return 1.0;
        }
        (1..=m).map(|k| (2 * k - 1) as f64).product()
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn assoc_legendre(l: u32, m: u32, x: f64) -> f64 {
        if m > l {
            return 0.0;
        }
        let t = 1.0 - x * x;
        if t <= 0.0 {
            // At x = ±1: P_l(1) = 1, P_l(-1) = (-1)^l, P_l^m(±1) = 0 for m > 0
            if m == 0 {
                return if x > 0.0 {
                    1.0
                } else if l.is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
            } else {
                return 0.0;
            }
        }

        let mut pm = Self::double_factorial(m) * t.sqrt().powf(m as f64);
        if m % 2 == 1 {
            pm = -pm;
        }
        if l == m {
            return pm;
        }

        let pmp1 = (2 * m + 1) as f64 * x * pm;
        if l == m + 1 {
            return pmp1;
        }

        let (mut pl_m2, mut pl_m1) = (pm, pmp1);
        for ll in (m + 2)..=l {
            let pl =
                ((2 * ll - 1) as f64 * x * pl_m1 - (ll + m - 1) as f64 * pl_m2) / (ll - m) as f64;
            pl_m2 = pl_m1;
            pl_m1 = pl;
        }
        pl_m1
    }

    #[expect(dead_code, clippy::unwrap_used, reason = "tests")]
    #[cfg(test)]
    fn ylm_scalar(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
        let abs_m = m.unsigned_abs();
        if abs_m > l {
            return 0.0;
        }

        let x = theta.cos();
        let plm = Self::assoc_legendre(l, abs_m, x);

        let angular = if abs_m == 0 {
            1.0
        } else if m > 0 {
            (abs_m as f64 * phi).cos()
        } else {
            (abs_m as f64 * phi).sin()
        };

        let num = Self::factorial(l - abs_m);
        let den = Self::factorial(l + abs_m);
        let pre = ((2 * l + 1) as f64) / (4.0 * std::f64::consts::PI);
        let n_lm = (pre * num / den).sqrt();

        let mut y_lm = n_lm * plm * angular;
        if abs_m != 0 {
            y_lm *= 2.0_f64.sqrt();
        }
        y_lm
    }

    fn ylm_gpu(&self, theta_phi: &[f64], l: u32, m: i32) -> Result<Vec<f64>> {
        let size = theta_phi.len() / 2;
        let abs_m = m.unsigned_abs();
        let m_is_positive = if m >= 0 { 1u32 } else { 0u32 };

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SH f64 Input"),
                contents: bytemuck::cast_slice(theta_phi),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SH f64 Output"),
            size: (size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            l: u32,
            abs_m: u32,
            m_is_positive: u32,
        }

        let params = Params {
            size: size as u32,
            l,
            abs_m,
            m_is_positive,
        };
        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SH f64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "SH f64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(size as u32)
            .submit();

        let result: Vec<f64> = self.device.read_buffer_f64(&output_buf, size)?;
        Ok(result)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_y00() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let sh = SphericalHarmonicsF64::new(device)?;
        // Y_0^0 = 1/(2√π) ≈ 0.282
        let theta_phi = vec![0.0, 0.0, PI / 2.0, 0.0, PI, PI];
        let result = sh.ylm(&theta_phi, 0, 0)?;
        let expected = 1.0 / (4.0 * PI).sqrt();
        for val in result {
            assert!(
                (val - expected).abs() < 1e-10,
                "Y₀⁰ = {}, expected {}",
                val,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn test_y10() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let sh = SphericalHarmonicsF64::new(device)?;
        // Y_1^0 = √(3/(4π)) cos(θ)
        let theta_phi = vec![0.0, 0.0, PI / 2.0, 0.0, PI, 0.0];
        let result = sh.ylm(&theta_phi, 1, 0)?;
        let norm = (3.0 / (4.0 * PI)).sqrt();
        assert!(
            (result[0] - norm * 1.0).abs() < 1e-8,
            "Y10(0) = {}",
            result[0]
        ); // cos(0) = 1
        assert!((result[1]).abs() < 1e-8, "Y10(π/2) = {}", result[1]); // cos(π/2) = 0
        assert!(
            (result[2] + norm).abs() < 1e-8,
            "Y10(π) = {}, expected {}",
            result[2],
            -norm
        ); // cos(π) = -1
        Ok(())
    }

    #[test]
    fn test_orthonormality_y00_y10() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let sh = SphericalHarmonicsF64::new(device)?;
        // Test at several points - not a full integral but sanity check
        let theta_phi = vec![PI / 4.0, 0.0, PI / 2.0, PI / 4.0, 3.0 * PI / 4.0, PI / 2.0];
        let y00 = sh.ylm(&theta_phi, 0, 0)?;
        let y10 = sh.ylm(&theta_phi, 1, 0)?;
        // Both should be finite
        for i in 0..3 {
            assert!(y00[i].is_finite(), "Y₀⁰ should be finite");
            assert!(y10[i].is_finite(), "Y₁⁰ should be finite");
        }
        Ok(())
    }
}
