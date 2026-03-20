// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bessel K₀ (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Modified Bessel function of third kind, order 0.
//! Applications: Yukawa potential, screened Coulomb, Green's functions

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/bessel_k0_f64.wgsl");
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
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, Arc<str>>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| Arc::from(format!("bessel_k0 DF64 rewrite failed: {e}")))
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

/// f64 Modified Bessel K0 function evaluator — pipeline-cached
pub struct BesselK0F64 {
    device: Arc<WgpuDevice>,
}

impl BesselK0F64 {
    /// Creates a new K₀ Bessel function evaluator for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute K₀(x) for each element (x > 0)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn k0(&self, x: &[f64]) -> Result<Vec<f64>> {
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
                label: Some("BesselK0 Input"),
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
            .create_uniform_buffer("BesselK0 Params", &metadata);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("BesselK0 BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("BesselK0 Pipeline"),
        );

        let workgroups = size.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BesselK0 Pass"),
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
    fn k0_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::k0_scalar(xi)).collect()
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn i0_small(x: f64) -> f64 {
        let y = x / 3.75;
        let t = y * y;
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    }

    #[allow(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
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
        assert!(result[0] > result[1], "K0 should decrease");
        assert!(result[1] > result[2], "K0 should decrease");
        Ok(())
    }
}
