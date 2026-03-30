// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clamp - Clamp values between min and max - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its min/max bounds
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Simple clamp variant (f64 canonical).
const WGSL_CLAMP_SIMPLE_F64: &str = include_str!("../shaders/math/clamp_simple_f64.wgsl");

/// Simple clamp variant (f32 derived from f64).
pub static WGSL_CLAMP_SIMPLE: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_CLAMP_SIMPLE_F64.to_string());

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/math/clamp_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Clamp operation - Clamp values between min and max
pub struct Clamp {
    input: Tensor,
    min_val: f32,
    max_val: f32,
}

impl Clamp {
    /// Create a new clamp operation
    #[must_use]
    pub fn new(input: Tensor, min_val: f32, max_val: f32) -> Self {
        Self {
            input,
            min_val,
            max_val,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the clamp operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            min_val: f32,
            max_val: f32,
        }

        let params = Params {
            size: size as u32,
            min_val: self.min_val,
            max_val: self.max_val,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Clamp Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Clamp Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Clamp Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
            label: Some("Clamp Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Clamp Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Clamp Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Clamp Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clamp Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Clamp values between min and max
    ///
    /// # Arguments
    ///
    /// * `min_val` - Minimum value
    /// * `max_val` - Maximum value
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn clamp_wgsl(self, min_val: f32, max_val: f32) -> Result<Self> {
        Clamp::new(self, min_val, max_val).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clamp() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![7], device);

        let output = input.clamp_wgsl(0.0, 2.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
        assert_eq!(result[5], 2.0);
        assert_eq!(result[6], 2.0);
    }

    #[tokio::test]
    async fn test_clamp_no_effect() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.5, 1.0, 1.5];
        let input = Tensor::new(data, vec![3], device);

        let output = input.clamp_wgsl(0.0, 2.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.5);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.5);
    }
}
