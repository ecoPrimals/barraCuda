// SPDX-License-Identifier: AGPL-3.0-only

//! Pairwise Hamming Distance — GPU kernel.
//!
//! Computes the upper-triangle pairwise Hamming distance matrix for N
//! sequences of length L. Each thread handles one pair. Output is
//! N*(N-1)/2 normalized distances (proportion of differing sites).
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;

static WGSL_PAIRWISE_HAMMING: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_str!("../../shaders/math/pairwise_hamming_f64.wgsl").to_string()
});

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HammingParams {
    n_seqs: u32,
    seq_len: u32,
}

/// GPU pairwise Hamming distance computation.
pub struct PairwiseHammingGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl PairwiseHammingGpu {
    /// Create pairwise Hamming distance calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PairwiseHamming BGL"),
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
            label: Some("PairwiseHamming Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PairwiseHamming Shader"),
            source: wgpu::ShaderSource::Wgsl((&*WGSL_PAIRWISE_HAMMING).into()),
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PairwiseHamming Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("pairwise_hamming"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute pairwise Hamming distances for `n_seqs` sequences of `seq_len`.
    ///
    /// `sequences_buf`: `[n_seqs × seq_len]` u32 (nucleotide codes)
    /// `distances_buf`: `[n_seqs*(n_seqs-1)/2]` f32 (normalized distances)
    pub fn dispatch(
        &self,
        sequences_buf: &wgpu::Buffer,
        distances_buf: &wgpu::Buffer,
        n_seqs: u32,
        seq_len: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = HammingParams { n_seqs, seq_len };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PairwiseHamming Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let n_pairs = n_seqs * (n_seqs - 1) / 2;

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PairwiseHamming BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sequences_buf.as_entire_binding(),
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
                label: Some("PairwiseHamming Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PairwiseHamming Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_pairs.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}
