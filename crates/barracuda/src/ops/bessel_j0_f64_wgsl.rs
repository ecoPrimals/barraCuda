// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bessel J₀ (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Bessel function of first kind, order 0.
//! Applications: cylindrical wave propagation, Fourier-Bessel transforms,
//! diffraction patterns, heat conduction in cylinders

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/bessel_j0_f64.wgsl");
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
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, String>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| format!("bessel_j0 DF64 rewrite failed: {e}"))
                });
            match DF64_RESULT.as_ref() {
                Ok(src) => Ok(src.as_str()),
                Err(msg) => Err(crate::error::BarracudaError::ShaderCompilation(msg.clone())),
            }
        }
    }
}

/// f64 Bessel J0 function evaluator — pipeline-cached
///
/// Computes J₀(x) using rational polynomial approximation
/// with full f64 precision.
pub struct BesselJ0F64 {
    device: Arc<WgpuDevice>,
}

impl BesselJ0F64 {
    /// Create new Bessel J0 f64 operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute J₀(x) for each element
    ///
    /// # Arguments
    /// * `x` - Input values
    ///
    /// # Returns
    /// Vector of J₀(x) values with f64 precision
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn j0(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_elementwise(x)
    }

    fn dispatch_elementwise(&self, x: &[f64]) -> Result<Vec<f64>> {
        let size = x.len();
        let ctx = get_device_context(&self.device);

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BesselJ0 Input"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(size);

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
        let params_buf = self
            .device
            .create_uniform_buffer("BesselJ0 Params", &metadata);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("BesselJ0 BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("BesselJ0 Pipeline"),
        );

        let workgroups = size.div_ceil(256) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BesselJ0 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, size)
    }

    #[allow(dead_code, reason = "CPU reference for GPU parity validation")]
    fn j0_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::j0_scalar(xi)).collect()
    }

    #[allow(dead_code, reason = "CPU reference for GPU parity validation")]
    fn j0_scalar(x: f64) -> f64 {
        let ax = x.abs();
        if ax >= 8.0 {
            let z = 8.0 / ax;
            let z2 = z * z;
            let z4 = z2 * z2;
            let z6 = z4 * z2;
            let z8 = z4 * z4;

            let pv = 1.0 - 1.098628627e-3 * z2 + 2.734510407e-5 * z4 - 2.073370639e-6 * z6
                + 2.093887211e-7 * z8;

            let qv = -1.562499995e-2 * z + 1.430488765e-4 * z * z2 - 6.911147651e-6 * z * z4
                + 7.621095161e-7 * z * z6
                - 9.349451520e-8 * z * z8;

            let sqrt_2_over_pi = 0.7978845608028654;
            let pi_over_4 = std::f64::consts::FRAC_PI_4;
            let inv_sqrt_x = sqrt_2_over_pi / ax.sqrt();
            let xx = ax - pi_over_4;
            inv_sqrt_x * (pv * xx.cos() - qv * xx.sin())
        } else {
            let z = x * x;
            let z2 = z * z;
            let z3 = z2 * z;
            let z4 = z2 * z2;
            let z5 = z2 * z3;

            let p = 57568490574.0 - 13362590354.0 * z + 651619640.7 * z2 - 11214424.18 * z3
                + 77392.33017 * z4
                - 184.9052456 * z5;

            let q = 57568490411.0
                + 1029532985.0 * z
                + 9494680.718 * z2
                + 59272.64853 * z3
                + 267.8532712 * z4
                + z5;

            p / q
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_j0_at_zero() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ0F64::new(device)?;

        let x = vec![0.0];
        let result = bessel.j0(&x)?;

        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "J₀(0) = {}, expected 1",
            result[0]
        );
        Ok(())
    }

    #[test]
    fn test_j0_known_values() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ0F64::new(device)?;

        let x = vec![1.0, 2.0, 5.0, 10.0];
        let expected = [
            0.7651976865579666,
            0.2238907791412357,
            -0.1775967713143383,
            -0.2459357644513483,
        ];

        let result = bessel.j0(&x)?;

        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "J₀({}) = {}, expected {}",
                x[i],
                val,
                expected[i]
            );
        }
        Ok(())
    }

    #[test]
    fn test_j0_symmetry() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ0F64::new(device)?;

        let x = vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        let result = bessel.j0(&x)?;

        assert!((result[0] - result[5]).abs() < 1e-10, "J₀(-3) != J₀(3)");
        assert!((result[1] - result[4]).abs() < 1e-10, "J₀(-2) != J₀(2)");
        assert!((result[2] - result[3]).abs() < 1e-10, "J₀(-1) != J₀(1)");
        Ok(())
    }

    #[test]
    fn test_j0_large() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bessel = BesselJ0F64::new(device)?;

        let x: Vec<f64> = (0..1000).map(|i| i as f64 * 0.02).collect();
        let result = bessel.j0(&x)?;

        assert_eq!(result.len(), 1000);
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "J₀(0) = {}, expected 1",
            result[0]
        );

        Ok(())
    }
}
