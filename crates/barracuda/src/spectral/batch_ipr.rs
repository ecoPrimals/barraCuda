// SPDX-License-Identifier: AGPL-3.0-only

//! Batch Inverse Participation Ratio (IPR) — GPU kernel.
//!
//! IPR measures eigenvector localization:
//!   IPR = Σ |ψ_i|⁴
//!
//! - Extended states: IPR ~ 1/dim
//! - Localized states: IPR >> 1/dim
//!
//! Each thread processes one eigenvector from a contiguous batch.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use crate::device::WgpuDevice;

pub static WGSL_BATCH_IPR: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/spectral/batch_ipr_f64.wgsl"
    ))
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct IprParams {
    dim: u32,
    n_vectors: u32,
}

pub struct BatchIprGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl BatchIprGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BatchIpr BGL"),
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
            label: Some("BatchIpr Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BatchIpr Shader"),
            source: wgpu::ShaderSource::Wgsl((&**WGSL_BATCH_IPR).into()),
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BatchIpr Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "batch_ipr",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute IPR for `n_vectors` eigenvectors, each of dimension `dim`.
    ///
    /// `eigenvectors_buf` layout: `[n_vectors × dim]` contiguous f32.
    /// Returns buffer of `[n_vectors]` f32 IPR values.
    pub fn dispatch(
        &self,
        eigenvectors_buf: &wgpu::Buffer,
        ipr_out_buf: &wgpu::Buffer,
        dim: u32,
        n_vectors: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = IprParams { dim, n_vectors };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchIpr Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchIpr BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: eigenvectors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ipr_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BatchIpr Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchIpr Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_vectors.div_ceil(256), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

use wgpu::util::DeviceExt;
