// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spatial Prisoner's Dilemma Payoff — GPU kernel.
//!
//! Computes cumulative payoff for each cell in a 2D grid using
//! Moore neighborhood (8 neighbors) with periodic boundary conditions.
//! Grid: 1 = cooperator, 0 = defector.
//!
//! Payoff rules:
//! - Both cooperate: b - c
//! - Cooperator exploited: -c
//! - Defector exploits: b
//! - Both defect: 0
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;

static WGSL_SPATIAL_PAYOFF: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_str!(
        "../../shaders/math/spatial_payoff_f64.wgsl"
    ).to_string()
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PayoffParams {
    grid_size: u32,
    b_x1000: u32,
    c_x1000: u32,
    _pad: u32,
}

/// GPU spatial payoff (evolutionary game) computation.
pub struct SpatialPayoffGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl SpatialPayoffGpu {
    /// Create spatial payoff calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpatialPayoff BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpatialPayoff Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpatialPayoff Shader"),
            source: wgpu::ShaderSource::Wgsl((&*WGSL_SPATIAL_PAYOFF).into()),
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialPayoff Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("spatial_payoff"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute spatial PD payoffs for a `grid_size × grid_size` grid.
    ///
    /// `grid_buf`: `[grid_size²]` u32 (0 = defector, 1 = cooperator)
    /// `fitness_buf`: `[grid_size²]` f32 (cumulative payoff)
    /// `benefit` / `cost`: PD parameters (encoded as x1000 integers internally)
    pub fn dispatch(
        &self,
        grid_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        grid_size: u32,
        benefit: f32,
        cost: f32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = PayoffParams {
            grid_size,
            b_x1000: (benefit * 1000.0) as u32,
            c_x1000: (cost * 1000.0) as u32,
            _pad: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialPayoff Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = grid_size * grid_size;

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpatialPayoff BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fitness_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("SpatialPayoff Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialPayoff Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}
