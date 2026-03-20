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
use crate::device::capabilities::WORKGROUP_SIZE_1D;

/// WGSL source for swarm NN forward pass (f32).
pub const WGSL_SWARM_NN_FORWARD: &str = include_str!("../../shaders/bio/swarm_nn_forward.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_SWARM_NN_FORWARD_F64: &str =
    include_str!("../../shaders/bio/swarm_nn_forward_f64.wgsl");

/// f64 is the canonical source — math is universal, precision is silicon.
static WGSL_SWARM_NN_SCORES_F64: &str = include_str!("../../shaders/bio/swarm_nn_scores_f64.wgsl");
/// Max activation output for `mean_reduce` chaining (Paper 015, L-009).
/// Outputs f32 scores per (controller, eval) — different from forward which outputs u32 actions.
pub static WGSL_SWARM_NN_SCORES: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_SWARM_NN_SCORES_F64.to_string());

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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl SwarmNnGpu {
    /// Create a new swarm NN GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SwarmNn BGL"),
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
            label: Some("SwarmNn Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader_f64(WGSL_SWARM_NN_FORWARD_F64, Some("SwarmNn f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SwarmNn Pipeline"),
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

    /// Run forward pass for swarm of neural network controllers.
    ///
    /// `weights_buf`: `[n_controllers × weights_per_ctrl]` f64
    /// `inputs_buf`: `[n_controllers × n_evals × input_dim]` f64
    /// `actions_buf`: `[n_controllers × n_evals]` u32
    pub fn dispatch(
        &self,
        weights_buf: &wgpu::Buffer,
        inputs_buf: &wgpu::Buffer,
        actions_buf: &wgpu::Buffer,
        params: &SwarmNnParams,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SwarmNn Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let total = params.n_controllers * params.n_evals;

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SwarmNn BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: actions_buf.as_entire_binding(),
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
                label: Some("SwarmNn Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SwarmNn Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(total.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
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
