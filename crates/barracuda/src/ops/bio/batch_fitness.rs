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
use crate::device::compute_pipeline::ComputeDispatch;

/// WGSL source for batch fitness evaluation (f32).
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
    device: Arc<WgpuDevice>,
}

impl BatchFitnessGpu {
    /// Create batch fitness GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Evaluate linear fitness for `pop_size` individuals, each with `genome_len` traits.
    ///
    /// `population_buf`: `[pop_size × genome_len]` f64 (row-major genotypes)
    /// `weights_buf`:    `[genome_len]` f64
    /// `fitness_buf`:    `[pop_size]` f64 (output)
    #[expect(
        clippy::missing_panics_doc,
        reason = "dispatch submit is infallible on valid device"
    )]
    pub fn dispatch(
        &self,
        population_buf: &wgpu::Buffer,
        weights_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        pop_size: u32,
        genome_len: u32,
    ) {
        let d = self.device.device();

        let params = FitnessParams {
            pop_size,
            genome_len,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchFitness Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        ComputeDispatch::new(&self.device, "BatchFitness")
            .shader(WGSL_BATCH_FITNESS_EVAL_F64, "batch_fitness_linear")
            .f64()
            .storage_read(0, population_buf)
            .storage_read(1, weights_buf)
            .storage_rw(2, fitness_buf)
            .uniform(3, &params_buf)
            .dispatch_1d(pop_size)
            .submit()
            .expect("BatchFitness GPU dispatch failed");
    }
}

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
        let _ = device.compile_shader_f64(WGSL_BATCH_FITNESS_EVAL_F64, Some("batch_fitness_f64"));
    }
}
