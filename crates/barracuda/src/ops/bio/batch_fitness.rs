// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch Fitness Evaluation — GPU kernel.
//!
//! Evaluates linear fitness for an entire evolutionary algorithm population
//! in a single GPU dispatch. Fitness is a dot product of genotype with
//! trait-weight vector.
//!
//! `fitness[i] = genotype[i] · weights`
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;

pub const WGSL_BATCH_FITNESS_EVAL: &str = include_str!("../../shaders/ml/batch_fitness_eval.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_BATCH_FITNESS_EVAL_F64: &str =
    include_str!("../../shaders/ml/batch_fitness_eval_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
}

/// Batch fitness evaluation GPU kernel (f64 pipeline).
pub struct BatchFitnessGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl BatchFitnessGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BatchFitness BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("BatchFitness Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let module =
            device.compile_shader_f64(WGSL_BATCH_FITNESS_EVAL_F64, Some("BatchFitness f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BatchFitness Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "batch_fitness_linear",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Evaluate linear fitness for `pop_size` individuals, each with `genome_len` traits.
    ///
    /// `population_buf`: `[pop_size × genome_len]` f64 (row-major genotypes)
    /// `weights_buf`:    `[genome_len]` f64
    /// `fitness_buf`:    `[pop_size]` f64 (output)
    pub fn dispatch(
        &self,
        population_buf: &wgpu::Buffer,
        weights_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        pop_size: u32,
        genome_len: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = FitnessParams {
            pop_size,
            genome_len,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchFitness Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchFitness BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: population_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fitness_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BatchFitness Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchFitness Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(pop_size.div_ceil(256), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_batch_fitness_linear() {
        assert!(WGSL_BATCH_FITNESS_EVAL_F64.contains("fn batch_fitness_linear"));
        assert!(WGSL_BATCH_FITNESS_EVAL_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        device.compile_shader_f64(WGSL_BATCH_FITNESS_EVAL_F64, Some("batch_fitness_f64"));
    }
}
