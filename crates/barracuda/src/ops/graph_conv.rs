// SPDX-License-Identifier: AGPL-3.0-or-later
//! GraphConv - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Graph Convolution operation
pub struct GraphConv {
    node_features: Tensor,
    adj_matrix: Tensor,
    weight: Tensor,
    bias: Tensor,
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
}

impl GraphConv {
    /// Create a new graph convolution operation
    pub fn new(
        node_features: Tensor,
        adj_matrix: Tensor,
        weight: Tensor,
        bias: Tensor,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        let num_nodes = node_shape[0];
        let in_features = node_shape[1..].iter().product::<usize>();

        let adj_shape = adj_matrix.shape();
        if adj_shape.len() != 2 || adj_shape[0] != num_nodes || adj_shape[1] != num_nodes {
            return Err(BarracudaError::invalid_op(
                "graph_conv",
                "adj_matrix must be [num_nodes, num_nodes]",
            ));
        }

        let weight_shape = weight.shape();
        let out_features = weight_shape[weight_shape.len() - 1];
        if weight_shape[0..weight_shape.len() - 1]
            .iter()
            .product::<usize>()
            != in_features
        {
            return Err(BarracudaError::invalid_op(
                "graph_conv",
                "weight shape mismatch",
            ));
        }

        let bias_size = bias.shape().iter().product::<usize>();
        if bias_size != out_features {
            return Err(BarracudaError::invalid_op(
                "graph_conv",
                "bias must have out_features elements",
            ));
        }

        Ok(Self {
            node_features,
            adj_matrix,
            weight,
            bias,
            num_nodes,
            in_features,
            out_features,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/gnn/graph_conv_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    /// Execute the graph convolution operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();

        // Create output buffer
        let output_size = self.num_nodes * self.out_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            in_features: u32,
            out_features: u32,
            _padding: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GraphConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("GraphConv Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GraphConv Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GraphConv Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.node_features.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.adj_matrix.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GraphConv Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GraphConv Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GraphConv Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GraphConv Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (self.num_nodes as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.out_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_graph_conv_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 3;
        let in_features = 4;
        let out_features = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let adj_matrix = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_nodes],
            vec![num_nodes, num_nodes],
            device.clone(),
        )
        .await
        .unwrap();

        let weight = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let bias = Tensor::from_vec_on(vec![0.0; out_features], vec![out_features], device.clone())
            .await
            .unwrap();

        let output = GraphConv::new(node_features, adj_matrix, weight, bias)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }
}
