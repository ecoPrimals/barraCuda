// SPDX-License-Identifier: AGPL-3.0-or-later

//! K-mer Histogram — GPU kernel.
//!
//! Computes a 4^k histogram from encoded k-mer sequences using atomic
//! increments. One thread per k-mer.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;

/// WGSL shader for k-mer histogram computation (atomic increments).
pub const WGSL_KMER_HISTOGRAM: &str = include_str!("../../shaders/bio/kmer_histogram.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KmerConfig {
    n_kmers: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU kernel for k-mer histogram: counts occurrences into 4^k bins.
pub struct KmerHistogramGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl KmerHistogramGpu {
    /// Create a k-mer histogram GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("KmerHistogram BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("KmerHistogram Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("KmerHistogram Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_KMER_HISTOGRAM.into()),
        });

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("KmerHistogram Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("kmer_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Count k-mer occurrences into a histogram.
    ///
    /// `kmers_buf`: `[n_kmers]` u32 — encoded k-mer hashes (each < 4^k)
    /// `histogram_buf`: `[4^k]` u32 — output histogram (must be zeroed before dispatch)
    pub fn dispatch(
        &self,
        kmers_buf: &wgpu::Buffer,
        histogram_buf: &wgpu::Buffer,
        n_kmers: u32,
        k: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let config = KmerConfig {
            n_kmers,
            k,
            _pad0: 0,
            _pad1: 0,
        };
        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KmerHistogram Config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KmerHistogram BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kmers_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: histogram_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("KmerHistogram Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("KmerHistogram Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_kmers.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contains_entry_point() {
        assert!(WGSL_KMER_HISTOGRAM.contains("fn kmer_histogram"));
    }
}
