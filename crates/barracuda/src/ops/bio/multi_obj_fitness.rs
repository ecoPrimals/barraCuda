// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-objective Fitness — GPU kernel.
//!
//! Evaluates per-individual multi-objective fitness from genotypes.
//! Reads genotypes [pop × genome_len] f64, writes fitness [pop × n_obj] f64.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;

pub const WGSL_MULTI_OBJ_FITNESS: &str = include_str!("../../shaders/bio/multi_obj_fitness.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_MULTI_OBJ_FITNESS_F64: &str =
    include_str!("../../shaders/bio/multi_obj_fitness_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MultiObjFitnessParams {
    pop: u32,
    genome_len: u32,
    n_obj: u32,
    _pad: u32,
}

/// Multi-objective fitness GPU kernel (f64 pipeline).
pub struct MultiObjFitnessGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl MultiObjFitnessGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MultiObjFitness BGL"),
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
            label: Some("MultiObjFitness Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module =
            device.compile_shader_f64(WGSL_MULTI_OBJ_FITNESS_F64, Some("MultiObjFitness f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MultiObjFitness Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute multi-objective fitness for `pop` genotypes of length `genome_len`.
    ///
    /// `genotypes_buf`: `[pop × genome_len]` f64
    /// `fitness_buf`: `[pop × n_obj]` f64
    pub fn dispatch(
        &self,
        genotypes_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        pop: u32,
        genome_len: u32,
        n_obj: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = MultiObjFitnessParams {
            pop,
            genome_len,
            n_obj,
            _pad: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MultiObjFitness Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = pop * n_obj;

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MultiObjFitness BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: genotypes_buf.as_entire_binding(),
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
                label: Some("MultiObjFitness Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MultiObjFitness Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_main() {
        assert!(WGSL_MULTI_OBJ_FITNESS_F64.contains("fn main"));
        assert!(WGSL_MULTI_OBJ_FITNESS_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        device.compile_shader_f64(WGSL_MULTI_OBJ_FITNESS_F64, Some("multi_obj_fitness_f64"));
    }
}
