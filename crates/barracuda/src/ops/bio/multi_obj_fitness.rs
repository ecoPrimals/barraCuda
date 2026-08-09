// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-objective Fitness — GPU kernel.
//!
//! Evaluates per-individual multi-objective fitness from genotypes.
//! Reads genotypes [pop × `genome_len`] f64, writes fitness [pop × `n_obj`] f64.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;

/// WGSL source for multi-objective fitness (f32).
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
    device: Arc<WgpuDevice>,
}

impl MultiObjFitnessGpu {
    /// Create multi-objective fitness GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute multi-objective fitness for `pop` genotypes of length `genome_len`.
    ///
    /// `genotypes_buf`: `[pop × genome_len]` f64
    /// `fitness_buf`: `[pop × n_obj]` f64
    #[expect(
        clippy::missing_panics_doc,
        reason = "dispatch submit is infallible on valid device"
    )]
    pub fn dispatch(
        &self,
        genotypes_buf: &wgpu::Buffer,
        fitness_buf: &wgpu::Buffer,
        pop: u32,
        genome_len: u32,
        n_obj: u32,
    ) {
        let d = self.device.device();

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

        ComputeDispatch::new(&self.device, "MultiObjFitness")
            .shader(WGSL_MULTI_OBJ_FITNESS_F64, "main")
            .f64()
            .storage_read(0, genotypes_buf)
            .storage_rw(1, fitness_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(total)
            .submit()
            .expect("MultiObjFitness GPU dispatch failed");
    }
}

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
        let _ =
            device.compile_shader_f64(WGSL_MULTI_OBJ_FITNESS_F64, Some("multi_obj_fitness_f64"));
    }
}
