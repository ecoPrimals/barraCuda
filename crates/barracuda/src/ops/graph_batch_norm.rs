// SPDX-License-Identifier: AGPL-3.0-only
//! Graph Batch Normalization - Batch normalization adapted for graph data (Pure WGSL)
//!
//! Normalizes node features across the batch and feature dimensions
//! Similar to standard batch norm, but operates on graph nodes
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Graph Batch Normalization
pub struct GraphBatchNorm {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_nodes: usize,
    num_features: usize,
    epsilon: f32,
}

impl GraphBatchNorm {
    /// Create graph batch normalization.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, gamma: Tensor, beta: Tensor, epsilon: f32) -> Result<Self> {
        let input_shape = input.shape();
        if input_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "input must be 2D [num_nodes, num_features]",
            ));
        }

        let num_nodes = input_shape[0];
        let num_features = input_shape[1];

        let gamma_shape = gamma.shape();
        let gamma_size = gamma_shape.iter().product::<usize>();
        if gamma_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "gamma must have num_features elements",
            ));
        }

        let beta_shape = beta.shape();
        let beta_size = beta_shape.iter().product::<usize>();
        if beta_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "beta must have num_features elements",
            ));
        }

        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "epsilon must be positive",
            ));
        }

        Ok(Self {
            input,
            gamma,
            beta,
            num_nodes,
            num_features,
            epsilon,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/norm/graph_batch_norm_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute graph batch normalization.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        // Create intermediate buffers for mean and variance
        let mean_buffer = device.create_buffer_f32(self.num_features)?;
        let variance_buffer = device.create_buffer_f32(self.num_features)?;

        // Create output buffer
        let output_size = self.num_nodes * self.num_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_features: u32,
            epsilon: f32,
            _pad1: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_features: self.num_features as u32,
            epsilon: self.epsilon,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GraphBatchNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("GraphBatchNorm Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GraphBatchNorm Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GraphBatchNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.gamma.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.beta.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mean_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: variance_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GraphBatchNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        // Create encoder
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GraphBatchNorm Encoder"),
        });

        // Step 1: Compute mean
        let mean_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GraphBatchNorm Mean Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("compute_mean"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GraphBatchNorm Mean Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&mean_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (self.num_features as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 2: Compute variance
        let variance_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GraphBatchNorm Variance Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("compute_variance"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GraphBatchNorm Variance Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&variance_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (self.num_features as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 3: Normalize, scale, and shift
        let normalize_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GraphBatchNorm Normalize Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("normalize"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GraphBatchNorm Normalize Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&normalize_pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.num_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_graph_batch_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 4;
        let num_features = 8;

        let input = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gamma =
            Tensor::from_vec_on(vec![1.0; num_features], vec![num_features], device.clone())
                .await
                .unwrap();

        let beta = Tensor::from_vec_on(vec![0.0; num_features], vec![num_features], device.clone())
            .await
            .unwrap();

        let batch_norm = GraphBatchNorm::new(input, gamma, beta, 1e-5).unwrap();
        let output = batch_norm.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, num_features]);
    }

    #[tokio::test]
    async fn test_graph_batch_norm_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 100;
        let num_features = 128;

        let input = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gamma =
            Tensor::from_vec_on(vec![1.0; num_features], vec![num_features], device.clone())
                .await
                .unwrap();

        let beta = Tensor::from_vec_on(vec![0.0; num_features], vec![num_features], device.clone())
            .await
            .unwrap();

        let batch_norm = GraphBatchNorm::new(input, gamma, beta, 1e-5).unwrap();
        let output = batch_norm.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, num_features]);
    }
}
