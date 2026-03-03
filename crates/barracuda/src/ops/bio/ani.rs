// SPDX-License-Identifier: AGPL-3.0-only

//! Batch pairwise Average Nucleotide Identity (ANI) on GPU.
//!
//! One thread per sequence pair. Counts identical non-gap bases across
//! alignment positions, producing ANI ∈ [0, 1].
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `ani_batch_f64.wgsl` — 7/7 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/ani_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AniParams {
    n_pairs: u32,
    max_seq_len: u32,
}

/// Batch ANI computation on GPU.
pub struct AniBatchF64 {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl AniBatchF64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("ani_batch_f64"));
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AniBatch:bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage(1, true),  // seq_a
                    bgl_storage(2, true),  // seq_b
                    bgl_storage(3, false), // ani_out
                    bgl_storage(4, false), // aligned_out
                    bgl_storage(5, false), // identical_out
                ],
            });
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("AniBatch:layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AniBatch:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    /// Dispatch ANI computation on GPU-resident buffers.
    pub fn dispatch(
        &self,
        n_pairs: u32,
        max_seq_len: u32,
        seq_a: &wgpu::Buffer,
        seq_b: &wgpu::Buffer,
        ani_out: &wgpu::Buffer,
        aligned_out: &wgpu::Buffer,
        identical_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = AniParams {
            n_pairs,
            max_seq_len,
        };
        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AniBatch:params"),
            size: std::mem::size_of::<AniParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("AniBatch:bg"),
                layout: &self.bgl,
                entries: &[
                    bg_entry(0, &params_buf),
                    bg_entry(1, seq_a),
                    bg_entry(2, seq_b),
                    bg_entry(3, ani_out),
                    bg_entry(4, aligned_out),
                    bg_entry(5, identical_out),
                ],
            });

        let mut enc = self
            .device
            .device
            .create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_pairs.div_ceil(256), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn bgl_storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn bg_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}
