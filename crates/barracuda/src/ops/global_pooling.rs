// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Pooling - Graph-level representation aggregation (Pure WGSL)
//!
//! Aggregate node features to graph-level representation
//! Supports: sum, mean, max aggregation
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Aggregation type for global pooling.
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    /// Sum over nodes.
    Sum,
    /// Mean over nodes.
    Mean,
    /// Max over nodes.
    Max,
}

/// Global pooling operation (graph-level aggregation).
pub struct GlobalPooling {
    node_features: Tensor,
    num_nodes: usize,
    num_features: usize,
    aggregation_type: AggregationType,
}

impl GlobalPooling {
    /// Create global pooling with the given aggregation type.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(node_features: Tensor, aggregation_type: AggregationType) -> Result<Self> {
        let node_shape = node_features.shape();
        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "global_pooling",
                "node_features must be 2D [num_nodes, num_features]",
            ));
        }

        let num_nodes = node_shape[0];
        let num_features = node_shape[1];

        Ok(Self {
            node_features,
            num_nodes,
            num_features,
            aggregation_type,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/global_pooling_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute global pooling and return the output tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        // Create output buffer
        let output_buffer = device.create_buffer_f32(self.num_features)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_features: u32,
            aggregation_type: u32,
            _pad1: u32,
        }

        let aggregation_code = match self.aggregation_type {
            AggregationType::Sum => 0u32,
            AggregationType::Mean => 1u32,
            AggregationType::Max => 2u32,
        };

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_features: self.num_features as u32,
            aggregation_type: aggregation_code,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GlobalPooling Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("GlobalPooling Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GlobalPooling Bind Group Layout"),
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
            label: Some("GlobalPooling Bind Group"),
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
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GlobalPooling Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        // Create pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GlobalPooling Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GlobalPooling Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GlobalPooling Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (self.num_features as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_global_pooling_sum() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 4;
        let num_features = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Sum).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_mean() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 3;
        let num_features = 4;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Mean).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_max() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 5;
        let num_features = 16;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Max).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let num_nodes = 100;
        let num_features = 128;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Mean).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }
}
