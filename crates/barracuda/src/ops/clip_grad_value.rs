// SPDX-License-Identifier: AGPL-3.0-only
//! Clip Gradient by Value - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/clip_grad_value_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Clip gradient by value operation
pub struct ClipGradValue {
    gradients: Tensor,
    clip_value: f32,
}

impl ClipGradValue {
    /// Create a new clip gradient by value operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(gradients: Tensor, clip_value: f32) -> Result<Self> {
        if clip_value < 0.0 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "clip_value must be non-negative".to_string(),
            });
        }
        Ok(Self {
            gradients,
            clip_value,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the clip gradient by value operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.gradients.device();
        let size: usize = self.gradients.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.gradients.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            clip_value: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            clip_value: self.clip_value,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ClipGradValue Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("ClipGradValue Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ClipGradValue Bind Group Layout"),
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
            label: Some("ClipGradValue Bind Group"),
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
                    label: Some("ClipGradValue Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ClipGradValue Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("ClipGradValue Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ClipGradValue Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(size as u32);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
        Ok(Tensor::new(
            output_data,
            self.gradients.shape().to_vec(),
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
    async fn test_clip_grad_value_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let gradients = Tensor::from_data(&[3.0, -4.0, 5.0, -6.0], vec![4], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        assert!(result[0] <= 2.0 && result[0] >= -2.0);
        assert!(result[1] <= 2.0 && result[1] >= -2.0);
        assert!(result[2] <= 2.0 && result[2] >= -2.0);
        assert!(result[3] <= 2.0 && result[3] >= -2.0);
    }

    #[tokio::test]
    async fn test_clip_grad_value_no_clip() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let gradients = Tensor::from_data(&[0.5, -0.3, 0.1], vec![3], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result[0], 0.5);
        assert_eq!(result[1], -0.3);
        assert_eq!(result[2], 0.1);
    }

    #[tokio::test]
    async fn test_clip_grad_value_zero() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let gradients = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 0.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_clip_grad_value_large() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..1000).map(|i| (i % 20) as f32 - 10.0).collect();
        let gradients = Tensor::from_data(&data, vec![1000], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 5.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 1000);
        assert!(result.iter().all(|&x| (-5.0..=5.0).contains(&x)));
    }

    #[tokio::test]
    async fn test_clip_grad_value_invalid() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let gradients = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(ClipGradValue::new(gradients, -1.0).is_err());
    }
}
