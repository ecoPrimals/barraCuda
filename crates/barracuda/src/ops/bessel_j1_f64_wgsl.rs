// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL J1 F64 - Bessel function of first kind, order 1 - f64 precision WGSL
//!
//! Deep Debt Principles apply. See bessel_j0_f64_wgsl.rs for details.
//!
//! Applications: Electromagnetic wave propagation, antenna patterns

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// f64 Bessel J1 function evaluator
pub struct BesselJ1F64 {
    device: Arc<WgpuDevice>,
}

impl BesselJ1F64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/bessel_j1_f64.wgsl")
    }

    /// Compute J₁(x) for each element
    pub fn j1(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }

        self.j1_gpu(x)
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn j1_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::j1_scalar(xi)).collect()
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn j1_scalar(x: f64) -> f64 {
        let ax = x.abs();
        if ax >= 8.0 {
            let z = 8.0 / ax;
            let z2 = z * z;
            let z4 = z2 * z2;
            let z6 = z4 * z2;
            let z8 = z4 * z4;

            let p1 = 1.0 + 1.83105e-3 * z2 - 3.516396496e-4 * z4 + 2.457520174e-5 * z6
                - 2.40337019e-6 * z8;

            let q1 = 4.687499995e-2 * z - 2.002690873e-4 * z * z2 + 8.449199096e-6 * z * z4
                - 8.8228987e-7 * z * z6
                + 1.057874120e-7 * z * z8;

            let sqrt_2_over_pi = 0.7978845608028654;
            let three_pi_over_4 = 2.356_194_490_192_345;
            let inv_sqrt_x = sqrt_2_over_pi / ax.sqrt();
            let xx = ax - three_pi_over_4;
            let r = inv_sqrt_x * (p1 * xx.cos() - q1 * xx.sin());
            if x < 0.0 {
                -r
            } else {
                r
            }
        } else {
            let z = x * x;
            let z2 = z * z;
            let z3 = z2 * z;
            let z4 = z2 * z2;
            let z5 = z2 * z3;

            let p = 72362614232.0 - 7895059235.0 * z + 242396853.1 * z2 - 2972611.439 * z3
                + 15704.48260 * z4
                - 30.16036606 * z5;

            let q = 144725228442.0
                + 2300535178.0 * z
                + 18583304.74 * z2
                + 99447.43394 * z3
                + 376.9991397 * z4
                + z5;

            x * (p / q)
        }
    }

    fn j1_gpu(&self, x: &[f64]) -> Result<Vec<f64>> {
        let size = x.len();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bessel J1 f64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bessel J1 f64 Output"),
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
                    label: Some("Bessel J1 f64 Metadata"),
                    contents: bytemuck::bytes_of(&metadata),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(self.device.as_ref(), "Bessel J1 f64")
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
    fn test_j1_at_zero() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ1F64::new(device)?;
        let result = bessel.j1(&[0.0])?;
        assert!(
            (result[0]).abs() < 1e-10,
            "J₁(0) = {}, expected 0",
            result[0]
        );
        Ok(())
    }

    #[test]
    fn test_j1_known_values() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ1F64::new(device)?;
        let x = vec![1.0, 2.0, 5.0];
        let expected = vec![0.4400505857449335, 0.5767248077568734, -0.3275791375914652];
        let result = bessel.j1(&x)?;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "J₁({}) = {}, expected {}",
                x[i],
                val,
                expected[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_j1_antisymmetry() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ1F64::new(device)?;
        let x = vec![-2.0, 2.0];
        let result = bessel.j1(&x)?;
        assert!((result[0] + result[1]).abs() < 1e-10, "J₁(-x) != -J₁(x)");
        Ok(())
    }
}
