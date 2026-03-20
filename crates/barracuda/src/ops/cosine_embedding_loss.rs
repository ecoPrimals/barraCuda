// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cosine embedding loss operation
//!
//! Measures similarity between embeddings using cosine similarity
//! Used in metric learning, face recognition, and contrastive learning

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CosineEmbeddingLossParams {
    size: u32,
    margin: f32,
    _padding: [u32; 2],
}

/// Cosine embedding loss operation
pub struct CosineEmbeddingLoss {
    input1: Tensor,
    input2: Tensor,
    label: Tensor,
    margin: f32,
}

impl CosineEmbeddingLoss {
    /// Create cosine embedding loss operation
    /// # Errors
    /// Returns [`Err`] if input1 and input2 shapes differ, or label is not scalar [1].
    pub fn new(input1: Tensor, input2: Tensor, label: Tensor, margin: f32) -> Result<Self> {
        if input1.shape() != input2.shape() {
            return Err(BarracudaError::invalid_op(
                "cosine_embedding_loss",
                format!(
                    "input1 shape {:?} must match input2 shape {:?}",
                    input1.shape(),
                    input2.shape()
                ),
            ));
        }

        if label.shape() != [1] {
            return Err(BarracudaError::invalid_op(
                "cosine_embedding_loss",
                format!("label must be scalar [1], got shape {:?}", label.shape()),
            ));
        }

        Ok(Self {
            input1,
            input2,
            label,
            margin,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/cosine_embedding_loss_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute cosine embedding loss on tensors
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input1.device();
        let size = self.input1.len();

        // Create output buffer (scalar loss)
        let output_buffer = device.create_buffer_f32(1)?;

        // Create params
        let params = CosineEmbeddingLossParams {
            size: size as u32,
            margin: self.margin,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CosineEmbeddingLoss Params"),
            size: std::mem::size_of::<CosineEmbeddingLossParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CosineEmbeddingLoss Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
            label: Some("CosineEmbeddingLoss Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input1.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.input2.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.label.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("CosineEmbeddingLoss"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CosineEmbeddingLoss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CosineEmbeddingLoss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("CosineEmbeddingLoss Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CosineEmbeddingLoss Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output tensor (scalar)
        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_cosine_embedding_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input1 = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let input2 = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let label = Tensor::from_vec_on(vec![1.0], vec![1], device)
            .await
            .unwrap();

        let output = CosineEmbeddingLoss::new(input1, input2, label, 0.5)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0] >= 0.0);
    }
}
