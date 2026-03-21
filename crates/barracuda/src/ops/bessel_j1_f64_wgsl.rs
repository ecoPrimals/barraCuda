// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bessel J₁ (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Bessel function of first kind, order 1.
//! Applications: electromagnetic wave propagation, antenna patterns

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/bessel_j1_f64.wgsl");
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
                        .map_err(|e| Arc::from(format!("bessel_j1 DF64 rewrite failed: {e}")))
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

/// f64 Bessel J1 function evaluator — pipeline-cached
pub struct BesselJ1F64 {
    device: Arc<WgpuDevice>,
}

impl BesselJ1F64 {
    /// Creates a new J₁ Bessel function evaluator for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute J₁(x) for each element
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn j1(&self, x: &[f64]) -> Result<Vec<f64>> {
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
                label: Some("BesselJ1 Input"),
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
            .create_uniform_buffer("BesselJ1 Params", &metadata);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("BesselJ1 BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("BesselJ1 Pipeline"),
        );

        let workgroups = size.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BesselJ1 Pass"),
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
    fn j1_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::j1_scalar(xi)).collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
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
            if x < 0.0 { -r } else { r }
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
        let expected = [0.4400505857449335, 0.5767248077568734, -0.3275791375914652];
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
