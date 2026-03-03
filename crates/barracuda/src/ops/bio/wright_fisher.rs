// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wright-Fisher drift + selection — one generation step on GPU.
//!
//! Each thread handles one locus across all populations:
//!   1. **Selection**: p' = p × w_A / (p × w_A + (1−p))
//!   2. **Drift**: Binomial(2N, p') via PRNG
//!
//! Uses inline xoshiro128** PRNG seeded per-thread from a persistent
//! `prng_state` buffer.
//!
//! **Papers**: 024 (pangenome selection), 025 (meta-population dynamics)
//!
//! **Provenance**: neuralSpring metalForge → toadStool absorption (Feb 2026)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;

/// f64 canonical — f32 derived via downcast_f64_to_f32 when needed.
pub const WGSL_WRIGHT_FISHER_F64: &str =
    include_str!("../../shaders/bio/wright_fisher_step_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WfParams {
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
    _pad: u32,
}

/// Wright-Fisher drift + selection GPU kernel (f64 pipeline).
pub struct WrightFisherGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl WrightFisherGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WrightFisher BGL"),
            entries: &[
                storage_entry(0, true),  // freq_in
                storage_entry(1, true),  // selection coefficients
                storage_entry(2, false), // freq_out
                storage_entry(3, false), // prng_state (read-write)
                uniform_entry(4),        // params
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WrightFisher Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let module = device.compile_shader_f64(WGSL_WRIGHT_FISHER_F64, Some("WrightFisher f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("WrightFisher Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "wright_fisher",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Dispatch one Wright-Fisher generation.
    ///
    /// `freq_in_buf`:    `[n_pops × n_loci]` f64 — allele frequencies
    /// `selection_buf`:  `[n_loci]` f64 — selection coefficients
    /// `freq_out_buf`:   `[n_pops × n_loci]` f64 — output frequencies
    /// `prng_state_buf`: `[n_pops × n_loci × 4]` u32 — PRNG state
    pub fn dispatch(
        &self,
        freq_in_buf: &wgpu::Buffer,
        selection_buf: &wgpu::Buffer,
        freq_out_buf: &wgpu::Buffer,
        prng_state_buf: &wgpu::Buffer,
        n_pops: u32,
        n_loci: u32,
        two_n: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = WfParams {
            n_pops,
            n_loci,
            two_n,
            _pad: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WrightFisher Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = n_pops * n_loci;
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WrightFisher BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: freq_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: selection_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: freq_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prng_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WrightFisher"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WrightFisher Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(total.div_ceil(256), 1, 1);
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
    fn f64_shader_contains_wright_fisher() {
        assert!(WGSL_WRIGHT_FISHER_F64.contains("fn wright_fisher"));
        assert!(WGSL_WRIGHT_FISHER_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        device.compile_shader_f64(WGSL_WRIGHT_FISHER_F64, Some("wright_fisher_f64"));
    }
}
