// SPDX-License-Identifier: AGPL-3.0-or-later

//! Swarm NN Forward — GPU kernel.
//!
//! Forward pass for a population of neural network controllers.
//! Reads weights [`n_controllers` × `weights_per_ctrl`] f64 and inputs
//! [`n_controllers` × `n_evals` × `input_dim`] f64, writes actions
//! [`n_controllers` × `n_evals`] u32.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;

/// WGSL source for swarm NN forward pass (f32).
pub const WGSL_SWARM_NN_FORWARD: &str = include_str!("../../shaders/bio/swarm_nn_forward.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_SWARM_NN_FORWARD_F64: &str =
    include_str!("../../shaders/bio/swarm_nn_forward_f64.wgsl");

/// f64 is the canonical source — math is universal, precision is silicon.
static WGSL_SWARM_NN_SCORES_F64: &str = include_str!("../../shaders/bio/swarm_nn_scores_f64.wgsl");
/// Max activation output for `mean_reduce` chaining (Paper 015, L-009).
/// Outputs f32 scores per (controller, eval) — different from forward which outputs u32 actions.
pub const WGSL_SWARM_NN_SCORES: &str = WGSL_SWARM_NN_SCORES_F64;

/// Parameters for swarm NN forward pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SwarmNnParams {
    /// Number of neural network controllers
    pub n_controllers: u32,
    /// Number of evaluations per controller
    pub n_evals: u32,
    /// Input dimension
    pub input_dim: u32,
    /// Hidden layer dimension
    pub hidden_dim: u32,
    /// Output dimension
    pub output_dim: u32,
    /// Padding for alignment
    pub _pad0: u32,
    /// Padding for alignment
    pub _pad1: u32,
    /// Padding for alignment
    pub _pad2: u32,
}

/// Swarm NN forward GPU kernel (f64 pipeline).
pub struct SwarmNnGpu {
    device: Arc<WgpuDevice>,
}

impl SwarmNnGpu {
    /// Create a new swarm NN GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Run forward pass for swarm of neural network controllers.
    ///
    /// `weights_buf`: `[n_controllers × weights_per_ctrl]` f64
    /// `inputs_buf`: `[n_controllers × n_evals × input_dim]` f64
    /// `actions_buf`: `[n_controllers × n_evals]` u32
    #[expect(
        clippy::missing_panics_doc,
        reason = "dispatch submit is infallible on valid device"
    )]
    pub fn dispatch(
        &self,
        weights_buf: &wgpu::Buffer,
        inputs_buf: &wgpu::Buffer,
        actions_buf: &wgpu::Buffer,
        params: &SwarmNnParams,
    ) {
        let d = self.device.device();

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SwarmNn Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = params.n_controllers * params.n_evals;

        ComputeDispatch::new(&self.device, "SwarmNn")
            .shader(WGSL_SWARM_NN_FORWARD_F64, "main")
            .f64()
            .storage_read(0, weights_buf)
            .storage_read(1, inputs_buf)
            .storage_rw(2, actions_buf)
            .uniform(3, &params_buf)
            .dispatch_1d(total)
            .submit()
            .expect("SwarmNn GPU dispatch failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_main() {
        assert!(WGSL_SWARM_NN_FORWARD_F64.contains("fn main"));
        assert!(WGSL_SWARM_NN_FORWARD_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _ = device.compile_shader_f64(WGSL_SWARM_NN_FORWARD_F64, Some("swarm_nn_forward_f64"));
    }
}
