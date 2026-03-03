// SPDX-License-Identifier: AGPL-3.0-or-later
//! BESSEL I0 F64 - Modified Bessel function of first kind, order 0 - f64 precision WGSL
//!
//! Deep Debt Principles apply. See bessel_j0_f64_wgsl.rs for details.
//!
//! Applications: Kaiser windows, cylindrical heat conduction, neutron diffusion

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// f64 Modified Bessel I0 function evaluator
pub struct BesselI0F64 {
    device: Arc<WgpuDevice>,
}

impl BesselI0F64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/bessel_i0_f64.wgsl")
    }

    /// Compute I₀(x) for each element
    pub fn i0(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.i0_gpu(x)
    }

    #[cfg(test)]
    fn i0_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::i0_scalar(xi)).collect()
    }

    #[cfg(test)]
    fn i0_scalar(x: f64) -> f64 {
        let ax = x.abs();
        if ax < 3.75 {
            let y = x / 3.75;
            let t = y * y;
            1.0 + t
                * (3.5156229
                    + t * (3.0899424
                        + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
        } else {
            let t = 3.75 / ax;
            let p = 0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
            ax.exp() * p / ax.sqrt()
        }
    }

    fn i0_gpu(&self, x: &[f64]) -> Result<Vec<f64>> {
        let size = x.len();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bessel I0 f64 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bessel I0 f64 Output"),
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
                    label: Some("Bessel I0 f64 Metadata"),
                    contents: bytemuck::bytes_of(&metadata),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(self.device.as_ref(), "Bessel I0 f64")
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
    fn test_i0_at_zero() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselI0F64::new(device)?;
        let result = bessel.i0(&[0.0])?;
        assert!(
            (result[0] - 1.0).abs() < 1e-10,
            "I₀(0) = {}, expected 1",
            result[0]
        );
        Ok(())
    }

    #[test]
    fn test_i0_known_values() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselI0F64::new(device)?;
        let x = vec![1.0, 2.0, 3.0];
        let expected = vec![1.2660658777520082, 2.2795853023360673, 4.880792585865024];
        let result = bessel.i0(&x)?;
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-5,
                "I₀({}) = {}, expected {}",
                x[i],
                val,
                expected[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_i0_symmetry() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselI0F64::new(device)?;
        let x = vec![-2.0, 2.0];
        let result = bessel.i0(&x)?;
        assert!((result[0] - result[1]).abs() < 1e-10, "I₀(-x) != I₀(x)");
        Ok(())
    }
}
