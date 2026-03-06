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
use crate::device::capabilities::WORKGROUP_SIZE_1D;

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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl StencilCooperationGpu {
    /// Create stencil cooperation kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StencilCoop BGL"),
            entries: &[
                storage_entry(0, true),  // strategies
                storage_entry(1, true),  // fitness
                storage_entry(2, false), // new_strategies
                uniform_entry(3),        // params
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StencilCoop Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module =
            device.compile_shader_f64(WGSL_STENCIL_COOPERATION_F64, Some("StencilCoop f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("StencilCoop Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("stencil_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Dispatch one imitation dynamics step.
    ///
    /// `strategies_buf`:     `[grid_size²]` u32 — current strategies
    /// `fitness_buf`:        `[grid_size²]` f64 — pre-computed fitness
    /// `new_strategies_buf`: `[grid_size²]` u32 — output strategies
    /// `kappa`:              selection intensity (temperature)
    /// `step`:               current generation (for neighbor rotation)
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
        let q = self.device.queue();

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
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StencilCoop BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: strategies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fitness_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_strategies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("StencilCoop"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("StencilCoop Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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
        device.compile_shader_f64(WGSL_STENCIL_COOPERATION_F64, Some("stencil_coop_f64"));
    }
}
