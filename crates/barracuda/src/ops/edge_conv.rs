// SPDX-License-Identifier: AGPL-3.0-or-later
//! Edge Convolution for Graph Neural Networks
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Learns edge features by aggregating neighbor information using CSR-format edges.
//!
//! Reference: "Dynamic Graph CNN for Learning on Point Clouds" by Wang et al. (2019)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeConvParams {
    num_nodes: u32,
    feature_dim: u32,
    output_dim: u32,
    num_edges: u32,
}

pub struct EdgeConv {
    node_features: Tensor,
    /// CSR row offsets: [num_nodes + 1] entries
    edge_offsets: Tensor,
    /// CSR column indices: [num_edges] neighbor node indices
    edge_targets: Tensor,
    mlp_weight: Tensor,
    mlp_bias: Tensor,
    num_edges: u32,
}

impl EdgeConv {
    /// Create EdgeConv operation with CSR-format edge storage
    ///
    /// # Arguments
    /// * `node_features` - Node features [num_nodes, feature_dim]
    /// * `edge_offsets` - CSR row offsets [num_nodes + 1] (stored as f32, cast to u32 in shader)
    /// * `edge_targets` - CSR column indices [num_edges] (stored as f32, cast to u32 in shader)
    /// * `mlp_weight` - MLP weight matrix [output_dim, 2 * feature_dim]
    /// * `mlp_bias` - MLP bias vector [output_dim]
    pub fn new(
        node_features: Tensor,
        edge_offsets: Tensor,
        edge_targets: Tensor,
        mlp_weight: Tensor,
        mlp_bias: Tensor,
    ) -> Result<Self> {
        let num_edges = edge_targets.len() as u32;

        Ok(Self {
            node_features,
            edge_offsets,
            edge_targets,
            mlp_weight,
            mlp_bias,
            num_edges,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/gnn/edge_conv_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    /// Execute EdgeConv on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        let node_shape = self.node_features.shape();

        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "EdgeConv",
                format!(
                    "node_features must be 2D [num_nodes, feature_dim], got shape {node_shape:?}"
                ),
            ));
        }

        let num_nodes = node_shape[0];
        let feature_dim = node_shape[1];
        let output_dim = self.mlp_bias.len();

        // Create output buffer: [num_nodes, output_dim]
        let output_size = num_nodes * output_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = EdgeConvParams {
            num_nodes: num_nodes as u32,
            feature_dim: feature_dim as u32,
            output_dim: output_dim as u32,
            num_edges: self.num_edges,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("EdgeConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout (7 bindings to match evolved WGSL)
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("EdgeConv Bind Group Layout"),
                    entries: &[
                        // binding 0: node_features (storage read)
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
                        // binding 1: edge_offsets (storage read)
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
                        // binding 2: edge_targets (storage read)
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
                        // binding 3: mlp_weight (storage read)
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
                        // binding 4: mlp_bias (storage read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 5: output (storage read_write)
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
                        // binding 6: params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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
            label: Some("EdgeConv Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.node_features.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.edge_offsets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.edge_targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.mlp_weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mlp_bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("EdgeConv"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("EdgeConv Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("EdgeConv Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("EdgeConv Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeConv Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (num_nodes as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_nodes, output_dim],
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_edge_conv_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 5;
        let feature_dim = 3;
        let output_dim = 4;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * feature_dim],
            vec![num_nodes, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        // CSR format: chain graph 0→1→2→3→4
        // edge_offsets: [0, 1, 2, 3, 4, 4] (node 4 has no outgoing edges)
        // edge_targets: [1, 2, 3, 4]
        let edge_offsets = Tensor::from_vec_on(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0], // num_nodes + 1 entries
            vec![num_nodes + 1],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_targets = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0], // 4 edges
            vec![4],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_weight = Tensor::from_vec_on(
            vec![0.1; output_dim * 2 * feature_dim],
            vec![output_dim, 2 * feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_bias = Tensor::from_vec_on(vec![0.0; output_dim], vec![output_dim], device.clone())
            .await
            .unwrap();

        let result = EdgeConv::new(
            node_features,
            edge_offsets,
            edge_targets,
            mlp_weight,
            mlp_bias,
        )
        .unwrap()
        .execute()
        .unwrap();

        assert_eq!(result.shape(), &[num_nodes, output_dim]);
    }
}
