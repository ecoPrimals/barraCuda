// SPDX-License-Identifier: AGPL-3.0-or-later
//! GINConv - Graph Isomorphism Network (Pure WGSL)
//!
//! Expressive GNN with MLP: h_i' = MLP((1 + ε) * h_i + Σ_j h_j)
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Graph Isomorphism Network Convolution
pub struct GinConv {
    node_features: Tensor,
    edge_index: Vec<(usize, usize)>,
    mlp_weights: Tensor,
    mlp_bias: Tensor,
    num_nodes: usize,
    num_edges: usize,
    in_features: usize,
    out_features: usize,
    epsilon: f32,
}

impl GinConv {
    pub fn new(
        node_features: Tensor,
        edge_index: Vec<(usize, usize)>,
        mlp_weights: Tensor,
        mlp_bias: Tensor,
        epsilon: f32,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "gin_conv",
                "node_features must be 2D [num_nodes, in_features]",
            ));
        }

        let num_nodes = node_shape[0];
        let in_features = node_shape[1];

        let weight_shape = mlp_weights.shape();
        if weight_shape.len() != 2 || weight_shape[0] != in_features {
            return Err(BarracudaError::invalid_op(
                "gin_conv",
                "mlp_weights must be [in_features, out_features]",
            ));
        }

        let out_features = weight_shape[1];

        let bias_shape = mlp_bias.shape();
        let bias_size = bias_shape.iter().product::<usize>();
        if bias_size != out_features {
            return Err(BarracudaError::invalid_op(
                "gin_conv",
                "mlp_bias must have out_features elements",
            ));
        }

        let num_edges = edge_index.len();
        if num_edges == 0 {
            return Err(BarracudaError::invalid_op(
                "gin_conv",
                "edge_index cannot be empty",
            ));
        }

        Ok(Self {
            node_features,
            edge_index,
            mlp_weights,
            mlp_bias,
            num_nodes,
            num_edges,
            in_features,
            out_features,
            epsilon,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/gnn/gin_conv_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        // Convert edge_index to u32 pairs
        let edge_data: Vec<u32> = self
            .edge_index
            .iter()
            .flat_map(|(src, dst)| vec![*src as u32, *dst as u32])
            .collect();

        // Create edge_index buffer
        let edge_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GINConv Edge Index"),
                contents: bytemuck::cast_slice(&edge_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create aggregated features buffer (zero-initialized for atomic accumulation)
        let aggregated_size = self.num_nodes * self.in_features;
        let aggregated_buffer = device.create_buffer_f32(aggregated_size)?;

        // Create output buffer
        let output_size = self.num_nodes * self.out_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_edges: u32,
            in_features: u32,
            out_features: u32,
            epsilon: f32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            epsilon: self.epsilon,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GINConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("GINConv Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GINConv Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            label: Some("GINConv Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.node_features.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.mlp_weights.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mlp_bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: aggregated_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GINConv Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create encoder
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GINConv Encoder"),
        });

        // Step 1: Aggregate neighbors
        let aggregate_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GINConv Aggregate Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "aggregate",
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GINConv Aggregate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&aggregate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (self.num_edges as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 2: Apply MLP
        let mlp_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GINConv MLP Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "apply_mlp",
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GINConv MLP Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&mlp_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (self.num_nodes as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.out_features],
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
    async fn test_gin_conv_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 4;
        let in_features = 8;
        let out_features = 16;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

        let mlp_weights = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_bias =
            Tensor::from_vec_on(vec![0.0; out_features], vec![out_features], device.clone())
                .await
                .unwrap();

        let gin = GinConv::new(node_features, edge_index, mlp_weights, mlp_bias, 0.0).unwrap();
        let output = gin.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_gin_conv_with_epsilon() {
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

        let edge_index = vec![(0, 1), (1, 2)];

        let mlp_weights = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_bias =
            Tensor::from_vec_on(vec![0.0; out_features], vec![out_features], device.clone())
                .await
                .unwrap();

        let gin = GinConv::new(node_features, edge_index, mlp_weights, mlp_bias, 0.1).unwrap();
        let output = gin.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_gin_conv_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 100;
        let in_features = 64;
        let out_features = 128;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let mut edge_index = Vec::new();
        for i in 0..num_nodes {
            edge_index.push((i, (i + 1) % num_nodes));
        }

        let mlp_weights = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_bias =
            Tensor::from_vec_on(vec![0.0; out_features], vec![out_features], device.clone())
                .await
                .unwrap();

        let gin = GinConv::new(node_features, edge_index, mlp_weights, mlp_bias, 0.0).unwrap();
        let output = gin.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }
}
