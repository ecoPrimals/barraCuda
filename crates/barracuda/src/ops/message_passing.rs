// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MessagePassing` - Pure WGSL
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

/// Message Passing operation
pub struct MessagePassing {
    node_features: Tensor,
    edge_index: Tensor,
    edge_features: Option<Tensor>,
    num_nodes: usize,
    num_edges: usize,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    message_dim: usize,
    aggr_type: u32, // 0 = sum, 1 = mean, 2 = max
}

impl MessagePassing {
    /// Create a new message passing operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `edge_index` is not [`num_edges`, 2].
    pub fn new(
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Option<Tensor>,
        aggr_type: u32,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        let num_nodes = node_shape[0];
        let node_feat_dim = node_shape[1..].iter().product::<usize>();

        let edge_shape = edge_index.shape();
        if edge_shape.len() != 2 || edge_shape[1] != 2 {
            return Err(BarracudaError::invalid_op(
                "message_passing",
                "edge_index must be [num_edges, 2]",
            ));
        }
        let num_edges = edge_shape[0];

        let edge_feat_dim = if let Some(ref ef) = edge_features {
            ef.shape()[1..].iter().product::<usize>()
        } else {
            0
        };

        Ok(Self {
            node_features,
            edge_index,
            edge_features,
            num_nodes,
            num_edges,
            node_feat_dim,
            edge_feat_dim,
            message_dim: node_feat_dim, // Simplified: message dim = node feat dim
            aggr_type,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/message_passing_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the message passing operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();

        // Create output buffer
        let output_size = self.num_nodes * self.node_feat_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_edges: u32,
            node_feat_dim: u32,
            edge_feat_dim: u32,
            message_dim: u32,
            aggr_type: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
            node_feat_dim: self.node_feat_dim as u32,
            edge_feat_dim: self.edge_feat_dim as u32,
            message_dim: self.message_dim as u32,
            aggr_type: self.aggr_type,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MessagePassing Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("MessagePassing Shader"));

        // Create bind group layout (simplified - edge_features optional)
        let mut entries = vec![
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
        ];

        // Add optional edge_features binding
        if self.edge_features.is_some() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        entries.extend([
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MessagePassing Bind Group Layout"),
                    entries: &entries,
                });

        // Create bind group entries
        let mut bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.node_features.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.edge_index.buffer().as_entire_binding(),
            },
        ];

        if let Some(ref ef) = self.edge_features {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: ef.buffer().as_entire_binding(),
            });
        }

        let placeholder = device.placeholder_buffer();
        bind_entries.extend([
            wgpu::BindGroupEntry {
                binding: 3,
                resource: placeholder.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: placeholder.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params_buffer.as_entire_binding(),
            },
        ]);

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MessagePassing Bind Group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MessagePassing Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MessagePassing Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("MessagePassing Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MessagePassing Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

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
            vec![self.num_nodes, self.node_feat_dim],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_message_passing_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 4;
        let num_edges = 3;
        let node_feat_dim = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * node_feat_dim],
            vec![num_nodes, node_feat_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = Tensor::from_vec_on(
            vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            vec![num_edges, 2],
            device.clone(),
        )
        .await
        .unwrap();

        let output = MessagePassing::new(node_features, edge_index, None, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[num_nodes, node_feat_dim]);
    }
}
