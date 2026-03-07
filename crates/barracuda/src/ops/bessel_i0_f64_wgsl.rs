// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bessel I₀ (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Modified Bessel function of first kind, order 0.
//! Applications: Kaiser windows, cylindrical heat conduction, neutron diffusion

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::error::Result;
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/bessel_i0_f64.wgsl");
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
                        .map_err(|e| format!("bessel_i0 DF64 rewrite failed: {e}"))
                });
            match DF64_RESULT.as_ref() {
                Ok(src) => Ok(src.as_str()),
                Err(msg) => Err(crate::error::BarracudaError::ShaderCompilation(msg.clone())),
            }
        }
    }
}

/// f64 Modified Bessel I0 function evaluator — pipeline-cached
pub struct BesselI0F64 {
    device: Arc<WgpuDevice>,
}

impl BesselI0F64 {
    /// Creates a new I₀ Bessel function evaluator for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute I₀(x) for each element
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn i0(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.is_empty() {
            return Ok(vec![]);
        }
        self.dispatch_elementwise(x)
    }

    fn dispatch_elementwise(&self, x: &[f64]) -> Result<Vec<f64>> {
        let size = x.len();
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BesselI0 Input"),
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
            .create_uniform_buffer("BesselI0 Params", &metadata);

        let layout_sig = BindGroupLayoutSignature::reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&input_buf, &output_buf, &params_buf],
            Some("BesselI0 BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            shader_src,
            layout_sig,
            "main",
            Some("BesselI0 Pipeline"),
        );

        let workgroups = size.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BesselI0 Pass"),
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
    fn i0_cpu(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| Self::i0_scalar(xi)).collect()
    }

    #[allow(dead_code, reason = "CPU scalar helper for GPU parity validation")]
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
        let expected = [1.2660658777520082, 2.2795853023360673, 4.880792585865024];
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
