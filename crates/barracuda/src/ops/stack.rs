// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stack - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Stack operation
pub struct Stack {
    tensors: Vec<Tensor>,
    dim: usize,
}

impl Stack {
    /// Create a new stack operation
    pub fn new(tensors: Vec<Tensor>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "Cannot stack empty tensor list".to_string(),
            });
        }

        // Validate all tensors have same shape
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "All tensors must have same shape. Tensor 0: {:?}, Tensor {}: {:?}",
                        first_shape,
                        i,
                        tensor.shape()
                    ),
                });
            }
        }

        if dim > first_shape.len() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("dim {} exceeds tensor rank {}", dim, first_shape.len()),
            });
        }

        Ok(Self { tensors, dim })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/stack_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the stack operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.tensors[0].device();
        let num_tensors = self.tensors.len();
        let tensor_size: usize = self.tensors[0].shape().iter().product();

        // Compute output shape
        let mut output_shape = self.tensors[0].shape().to_vec();
        output_shape.insert(self.dim, num_tensors);
        let output_size: usize = output_shape.iter().product();

        // Create concatenated input buffer using direct buffer-to-buffer copies
        let input_size = num_tensors * tensor_size;
        let input_buffer = device.create_buffer_f32(input_size)?;

        // Copy each tensor buffer directly to the concatenated buffer
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Stack Copy Encoder"),
        });

        for (i, tensor) in self.tensors.iter().enumerate() {
            let offset = i * tensor_size * std::mem::size_of::<f32>();
            encoder.copy_buffer_to_buffer(
                tensor.buffer(),
                0,
                &input_buffer,
                offset as u64,
                (tensor_size * std::mem::size_of::<f32>()) as u64,
            );
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StackParams {
            num_tensors: u32,
            tensor_size: u32,
            output_size: u32,
            stack_dim: u32,
        }

        let params = StackParams {
            num_tensors: num_tensors as u32,
            tensor_size: tensor_size as u32,
            output_size: output_size as u32,
            stack_dim: self.dim as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Stack Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Stack Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Stack Bind Group Layout"),
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
            label: Some("Stack Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
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
                    label: Some("Stack Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Stack Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Stack Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Stack Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_stack_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0], vec![2], device.clone()).unwrap();

        let stacked = Stack::new(vec![t1, t2], 0).unwrap().execute().unwrap();
        assert_eq!(stacked.shape(), &vec![2, 2]);
    }

    #[tokio::test]
    async fn test_stack_multiple() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let tensors: Vec<Tensor> = (0..5)
            .map(|i| Tensor::from_data(&[i as f32; 4], vec![2, 2], device.clone()).unwrap())
            .collect();

        let stacked = Stack::new(tensors, 0).unwrap().execute().unwrap();
        assert_eq!(stacked.shape(), &vec![5, 2, 2]);
    }

    #[tokio::test]
    async fn test_stack_empty() {
        let _device = get_test_device().await;
        assert!(Stack::new(vec![], 0).is_err());
    }

    #[tokio::test]
    async fn test_stack_shape_mismatch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0, 5.0], vec![3], device.clone()).unwrap();

        assert!(Stack::new(vec![t1, t2], 0).is_err());
    }

    #[tokio::test]
    async fn test_stack_dim_invalid() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0], vec![2], device.clone()).unwrap();

        assert!(Stack::new(vec![t1, t2], 10).is_err());
    }
}
