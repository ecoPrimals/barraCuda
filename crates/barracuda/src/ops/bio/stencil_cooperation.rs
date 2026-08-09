// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fermi imitation dynamics on 2D grid — stencil cooperation update.
//!
//! Each cell compares its fitness with a neighbor's via the Fermi function:
//!   P(adopt) = 1 / (1 + `exp((f_self` - `f_neighbor`) / κ))
//!
//! This is the standard imitation dynamics update rule for spatial
//! evolutionary game theory (Paper 019).
//!
//! **Requires**: fitness values pre-computed by [`super::spatial_payoff`].
//!
//! **Provenance**: neuralSpring metalForge → toadStool absorption (Feb 2026)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;

/// WGSL source for stencil cooperation (f32).
pub const WGSL_STENCIL_COOPERATION: &str =
    include_str!("../../shaders/bio/stencil_cooperation.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_STENCIL_COOPERATION_F64: &str =
    include_str!("../../shaders/bio/stencil_cooperation_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct StencilParams {
    grid_size: u32,
    kappa_x1000: u32,
    step: u32,
    _pad: u32,
}

/// Fermi imitation dynamics GPU kernel (f64 pipeline).
///
/// Updates strategy grid based on fitness comparison with Moore neighbors.
pub struct StencilCooperationGpu {
    device: Arc<WgpuDevice>,
}

impl StencilCooperationGpu {
    /// Create stencil cooperation kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Dispatch one imitation dynamics step.
    ///
    /// `strategies_buf`:     `[grid_size²]` u32 — current strategies
    /// `fitness_buf`:        `[grid_size²]` f64 — pre-computed fitness
    /// `new_strategies_buf`: `[grid_size²]` u32 — output strategies
    /// `kappa`:              selection intensity (temperature)
    /// `step`:               current generation (for neighbor rotation)
    #[expect(
        clippy::missing_panics_doc,
        reason = "dispatch submit is infallible on valid device"
    )]
    pub fn dispatch(
        &self,
        strategies_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        new_strategies_buf: &wgpu::Buffer,
        grid_size: u32,
        kappa: f64,
        step: u32,
    ) {
        let d = self.device.device();

        let params = StencilParams {
            grid_size,
            kappa_x1000: (kappa * 1000.0) as u32,
            step,
            _pad: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("StencilCoop Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = grid_size * grid_size;

        ComputeDispatch::new(&self.device, "StencilCoop")
            .shader(WGSL_STENCIL_COOPERATION_F64, "stencil_update")
            .f64()
            .storage_read(0, strategies_buf)
            .storage_read(1, fitness_buf)
            .storage_rw(2, new_strategies_buf)
            .uniform(3, &params_buf)
            .dispatch_1d(total)
            .submit()
            .expect("StencilCoop GPU dispatch failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_stencil_update() {
        assert!(WGSL_STENCIL_COOPERATION_F64.contains("fn stencil_update"));
        assert!(WGSL_STENCIL_COOPERATION_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _ = device.compile_shader_f64(WGSL_STENCIL_COOPERATION_F64, Some("stencil_coop_f64"));
    }
}
