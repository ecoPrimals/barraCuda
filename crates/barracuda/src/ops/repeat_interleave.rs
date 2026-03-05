// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repeat Interleave - Pure WGSL
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

/// Repeat interleave operation
pub struct RepeatInterleave {
    input: Tensor,
    repeats: usize,
    dim: usize,
}

impl RepeatInterleave {
    /// Create a new repeat interleave operation
    pub fn new(input: Tensor, repeats: usize, dim: usize) -> Result<Self> {
        let shape = input.shape();
        if dim >= shape.len() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("dim {} exceeds tensor rank {}", dim, shape.len()),
            });
        }

        if repeats == 0 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "repeats must be positive".to_string(),
            });
        }

        Ok(Self {
            input,
            repeats,
            dim,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/repeat_interleave_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the repeat interleave operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let dim_size = shape[self.dim];

        // Compute output shape
        let mut output_shape = shape.to_vec();
        output_shape[self.dim] = dim_size * self.repeats;
        let output_size: usize = output_shape.iter().product();
        let input_size: usize = shape.iter().product();

        // Compute inner and outer sizes
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let outer_size: usize = shape[..self.dim].iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            input_size: u32,
            repeats: u32,
            dim: u32,
            dim_size: u32,
            inner_size: u32,
            outer_size: u32,
            _pad1: u32,
        }

        let params = Params {
            output_size: output_size as u32,
            input_size: input_size as u32,
            repeats: self.repeats as u32,
            dim: self.dim as u32,
            dim_size: dim_size as u32,
            inner_size: inner_size as u32,
            outer_size: outer_size as u32,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RepeatInterleave Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("RepeatInterleave Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RepeatInterleave Bind Group Layout"),
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
            label: Some("RepeatInterleave Bind Group"),
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

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RepeatInterleave Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("RepeatInterleave Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RepeatInterleave Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RepeatInterleave Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_repeat_interleave_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device.clone()).unwrap();

        let result = RepeatInterleave::new(input, 2, 0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![6]);
    }

    #[tokio::test]
    async fn test_repeat_interleave_2d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3], device.clone()).unwrap();

        let result = RepeatInterleave::new(input, 3, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![2, 9]);
    }

    #[tokio::test]
    async fn test_repeat_interleave_invalid_dim() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();

        assert!(RepeatInterleave::new(input, 2, 10).is_err());
    }

    #[tokio::test]
    async fn test_repeat_interleave_zero() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();

        assert!(RepeatInterleave::new(input, 0, 0).is_err());
    }

    #[tokio::test]
    async fn test_repeat_interleave_large() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![10, 10], device.clone()).unwrap();

        let result = RepeatInterleave::new(input, 5, 0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![50, 10]);
    }
}
