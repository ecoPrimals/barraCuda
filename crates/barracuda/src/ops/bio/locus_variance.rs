// SPDX-License-Identifier: AGPL-3.0-only

//! Per-Locus Allele Frequency Variance — GPU kernel.
//!
//! Computes population variance of allele frequencies across populations
//! for each locus independently. Core building block for Weir-Cockerham
//! FST estimation.
//!
//! Input:  `allele_freqs[pop * n_loci + locus]`
//! Output: `per_locus_var[locus]`
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;

/// f64 canonical — f32 derived via downcast_f64_to_f32 when needed.
pub const WGSL_LOCUS_VARIANCE_F64: &str = include_str!("../../shaders/bio/locus_variance_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
}

/// Per-locus allele frequency variance GPU kernel (f64 pipeline).
pub struct LocusVarianceGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl LocusVarianceGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LocusVariance BGL"),
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
            label: Some("LocusVariance Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let module = device.compile_shader_f64(WGSL_LOCUS_VARIANCE_F64, Some("LocusVariance f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LocusVariance Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "locus_variance",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute per-locus allele frequency variance across populations.
    ///
    /// `allele_freqs_buf`: `[n_pops × n_loci]` f64
    /// `output_buf`:       `[n_loci]` f64
    pub fn dispatch(
        &self,
        allele_freqs_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        n_pops: u32,
        n_loci: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = VarianceParams { n_pops, n_loci };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LocusVariance Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LocusVariance BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: allele_freqs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LocusVariance Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LocusVariance Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n_loci.div_ceil(256), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_locus_variance() {
        assert!(WGSL_LOCUS_VARIANCE_F64.contains("fn locus_variance"));
        assert!(WGSL_LOCUS_VARIANCE_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        device.compile_shader_f64(WGSL_LOCUS_VARIANCE_F64, Some("locus_variance_f64"));
    }
}
