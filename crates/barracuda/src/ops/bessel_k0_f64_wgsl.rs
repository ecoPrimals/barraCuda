// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL K0 F64 - Modified Bessel function of third kind, order 0 - f64 precision WGSL
//!
//! Deep Debt Principles apply. See bessel_j0_f64_wgsl.rs for details.
//!
//! Applications: Yukawa potential, screened Coulomb, Green's functions

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// f64 Modified Bessel K0 function evaluator
pub struct BesselK0F64 {
    device: Arc<WgpuDevice>,
}

impl BesselK0F64 {
    /// Creates a new K₀ Bessel function evaluator for the given WGPU device.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/bessel_k0_f64.wgsl")
    }

    /// Compute K₀(x) for each element (x > 0)
    pub fn k0(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.k0_gpu(x)
    }

    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn k0_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::k0_scalar(xi)).collect()
    }

    #[cfg(test)]
    fn i0_small(x: f64) -> f64 {
        let y = x / 3.75;
        let t = y * y;
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    }

    #[cfg(test)]
    fn k0_scalar(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }
        if x <= 2.0 {
            let y = x * 0.5;
            let z = y * y;
            let p = -0.57721566
                + z * (0.42278420
                    + z * (0.23069756
                        + z * (0.03488590 + z * (0.00262698 + z * (0.00010750 + z * 0.00000740)))));
            p - Self::i0_small(x) * y.ln()
        } else {
            let t = 2.0 / x;
            let p = 1.25331414
                + t * (-0.07832358
                    + t * (0.02189568
                        + t * (-0.01062446
                            + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208)))));
            (-x).exp() * p / x.sqrt()
        }
    }

    fn k0_gpu(&self, x: &[f64]) -> Result<Vec<f64>> {
        let size = x.len();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bessel K0 f64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bessel K0 f64 Output"),
            size: std::mem::size_of_val(x) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Metadata {
            size: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let metadata = Metadata {
            size: size as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let metadata_buf =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Bessel K0 f64 Metadata"),
                    contents: bytemuck::bytes_of(&metadata),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(self.device.as_ref(), "Bessel K0 f64")
            .shader(Self::wgsl_shader(), "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &output_buf)
            .uniform(2, &metadata_buf)
            .dispatch_1d(size as u32)
            .submit();

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
    fn test_k0_known_values() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselK0F64::new(device)?;
        let x = vec![0.5, 1.0, 2.0, 5.0];
        let expected = [
            0.9244190712276659,
            0.4210244382407084,
            0.1138938727495334,
            0.003691098334042594,
        ];
        let result = bessel.k0(&x)?;
        for (i, &val) in result.iter().enumerate() {
            let rel_err = (val - expected[i]).abs() / expected[i];
            assert!(
                rel_err < 1e-5,
                "K₀({}) = {}, expected {}",
                x[i],
                val,
                expected[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_k0_exponential_decay() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselK0F64::new(device)?;
        let x = vec![5.0, 10.0, 15.0];
        let result = bessel.k0(&x)?;
        // K0 decays exponentially
        assert!(result[0] > result[1], "K0 should decrease");
        assert!(result[1] > result[2], "K0 should decrease");
        Ok(())
    }
}
