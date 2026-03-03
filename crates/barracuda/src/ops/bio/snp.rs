// SPDX-License-Identifier: AGPL-3.0-only

//! Position-parallel SNP calling on GPU.
//!
//! One thread per alignment column. Each thread counts allele frequencies
//! across all sequences, determines the reference allele (most common),
//! and flags polymorphic positions.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `snp_calling_f64.wgsl` — 5/5 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/snp_calling_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SnpParams {
    alignment_length: u32,
    n_sequences: u32,
    min_depth: u32,
    _pad: u32,
}

/// Position-parallel SNP calling on GPU.
pub struct SnpCallingF64 {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl SnpCallingF64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("snp_calling_f64"));
        let bgl = make_bgl(&device, &[true, false, false, false, false]);
        let layout = make_layout(&device, &bgl, "SnpCalling");
        let pipeline = make_pipeline(&device, &layout, &module, "main", "SnpCalling");
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    pub fn dispatch(
        &self,
        alignment_length: u32,
        n_sequences: u32,
        min_depth: u32,
        sequences: &wgpu::Buffer,
        is_variant: &wgpu::Buffer,
        ref_allele: &wgpu::Buffer,
        depth_out: &wgpu::Buffer,
        alt_freq_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = SnpParams {
            alignment_length,
            n_sequences,
            min_depth,
            _pad: 0,
        };
        let pbuf = upload_uniform(&self.device, &params);
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    bg_entry(0, &pbuf),
                    bg_entry(1, sequences),
                    bg_entry(2, is_variant),
                    bg_entry(3, ref_allele),
                    bg_entry(4, depth_out),
                    bg_entry(5, alt_freq_out),
                ],
            });
        submit(
            &self.device,
            &self.pipeline,
            &bg,
            alignment_length.div_ceil(256),
        );
        Ok(())
    }
}

pub(super) fn make_bgl(device: &WgpuDevice, storage_ro: &[bool]) -> wgpu::BindGroupLayout {
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    for (i, &ro) in storage_ro.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (i + 1) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    device
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        })
}

pub(super) fn make_layout(
    device: &WgpuDevice,
    bgl: &wgpu::BindGroupLayout,
    label: &str,
) -> wgpu::PipelineLayout {
    device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        })
}

pub(super) fn make_pipeline(
    device: &WgpuDevice,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: entry,
            compilation_options: Default::default(),
            cache: None,
        })
}

pub(super) fn upload_uniform<T: bytemuck::Pod>(device: &WgpuDevice, data: &T) -> wgpu::Buffer {
    let buf = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device.queue.write_buffer(&buf, 0, bytemuck::bytes_of(data));
    buf
}

pub(super) fn bg_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

pub(super) fn submit(
    device: &WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    wg_x: u32,
) {
    let mut enc = device.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(wg_x, 1, 1);
    }
    device.submit_and_poll(Some(enc.finish()));
}
