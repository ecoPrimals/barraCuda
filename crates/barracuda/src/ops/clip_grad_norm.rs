// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clip Gradient by Norm - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Two-pass operation:
//! 1. Compute total norm (parallel reduction)
//! 2. Clip gradients based on computed norm
//!
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/gradient/clip_grad_norm_f64.wgsl");

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Clip gradient by norm operation
pub struct ClipGradNorm {
    gradients: Tensor,
    max_norm: f32,
}

impl ClipGradNorm {
    /// Create a new clip gradient by norm operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(gradients: Tensor, max_norm: f32) -> Result<Self> {
        if max_norm < 0.0 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "max_norm must be non-negative".to_string(),
            });
        }
        Ok(Self {
            gradients,
            max_norm,
        })
    }

    /// Get the WGSL shader source (f64 canonical, downcast to f32 at compile)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| SHADER_F64.to_string());
        SHADER.as_str()
    }

    /// Execute the clip gradient by norm operation (2-pass)
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

        // Norm buffer: stores per-workgroup partial sums, then final norm in [0]
        // Size must be >= num_workgroups for pass 1, and has element [0] for clip pass
        let caps = DeviceCapabilities::from_device(device);
        let num_workgroups = caps.dispatch_1d(size as u32);
        let norm_buffer_size = num_workgroups.max(1) as usize;
        let norm_buffer = device.create_buffer_f32(norm_buffer_size)?;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            max_norm: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            max_norm: self.max_norm,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ClipGradNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("ClipGradNorm Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ClipGradNorm Bind Group Layout"),
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
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ClipGradNorm Bind Group"),
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
                    resource: norm_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ClipGradNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline_norm =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ClipGradNorm Norm Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("compute_norm"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let compute_pipeline_norm_final =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ClipGradNorm Norm Final Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("compute_norm_final"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let compute_pipeline_clip =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ClipGradNorm Clip Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("clip_gradients"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shaders
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("ClipGradNorm Encoder"),
        });

        // Pass 1: Compute per-workgroup sum of squares (workgroup tree reduction)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ClipGradNorm Norm Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline_norm);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Pass 2: Reduce partial sums to norm_buffer[0] (when multiple workgroups)
        if num_workgroups > 1 {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ClipGradNorm Norm Final Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline_norm_final);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 3: Clip gradients based on computed norm
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ClipGradNorm Clip Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline_clip);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            let workgroups = caps.dispatch_1d(size as u32);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            self.gradients.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Arc<crate::device::WgpuDevice> {
        crate::device::test_pool::get_test_device().await
    }

    #[tokio::test]
    async fn test_clip_grad_norm_basic() {
        let device = get_test_device().await;
        let gradients = Tensor::from_data(&[3.0, 4.0], vec![2], device.clone()).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        // Original norm = 5, should be clipped to norm = 1
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1); // Allow some tolerance for atomic operations
    }

    #[tokio::test]
    async fn test_clip_grad_norm_no_clip() {
        let device = get_test_device().await;
        let gradients = Tensor::from_data(&[0.1, 0.2, 0.3], vec![3], device.clone()).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        // Norm ≈ 0.374, should not be clipped
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0);
    }

    #[tokio::test]
    async fn test_clip_grad_norm_zero() {
        let device = get_test_device().await;
        let gradients = Tensor::from_data(&[0.0, 0.0, 0.0], vec![3], device.clone()).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_clip_grad_norm_large() {
        let device = get_test_device().await;
        let data: Vec<f32> = (0..1000).map(|i| (i % 10) as f32).collect();
        let gradients = Tensor::from_data(&data, vec![1000], device.clone()).unwrap();

        let clipped = ClipGradNorm::new(gradients, 100.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 1000);
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 100.0 + 1.0); // Allow tolerance
    }

    #[tokio::test]
    async fn test_clip_grad_norm_invalid() {
        let device = get_test_device().await;
        let gradients = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();

        assert!(ClipGradNorm::new(gradients, -1.0).is_err());
    }
}
