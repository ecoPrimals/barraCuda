// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pairwise Jaccard Distance — GPU kernel.
//!
//! Computes the upper-triangle Jaccard distance matrix for a pangenome
//! presence/absence (PA) matrix. Each thread handles one genome pair.
//!
//! Jaccard(i,j) = 1 - |intersection| / |union|
//!
//! PA matrix stored column-major: `pa[gene * n_genomes + genome]`.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;

static WGSL_PAIRWISE_JACCARD: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_str!("../../shaders/math/pairwise_jaccard_f64.wgsl").to_string()
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

/// GPU pairwise Jaccard similarity computation.
pub struct PairwiseJaccardGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl PairwiseJaccardGpu {
    /// Create pairwise Jaccard similarity calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PairwiseJaccard BGL"),
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
            label: Some("PairwiseJaccard Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PairwiseJaccard Shader"),
            source: wgpu::ShaderSource::Wgsl((&*WGSL_PAIRWISE_JACCARD).into()),
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PairwiseJaccard Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("pairwise_jaccard"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute pairwise Jaccard distances for a pangenome PA matrix.
    ///
    /// `pa_buf`: `[n_genes × n_genomes]` f32, column-major (1.0 = present, 0.0 = absent)
    /// `distances_buf`: `[n_genomes*(n_genomes-1)/2]` f32
    pub fn dispatch(
        &self,
        pa_buf: &wgpu::Buffer,
        distances_buf: &wgpu::Buffer,
        n_genomes: u32,
        n_genes: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = JaccardParams { n_genomes, n_genes };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PairwiseJaccard Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let n_pairs = n_genomes * (n_genomes - 1) / 2;

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PairwiseJaccard BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: distances_buf.as_entire_binding(),
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
                label: Some("PairwiseJaccard Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PairwiseJaccard Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_pairs.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}
