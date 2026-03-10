// SPDX-License-Identifier: AGPL-3.0-only
//! Threshold - Threshold activation function - Pure WGSL
//!
//! f64 canonical — f32 derived via `downcast_f64_to_f32` when needed.
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its threshold and value
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../shaders/activation/threshold_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Threshold operation
pub struct Threshold {
    input: Tensor,
    threshold: f32,
    value: f32,
}

impl Threshold {
    /// Create a new threshold operation
    #[must_use]
    pub fn new(input: Tensor, threshold: f32, value: f32) -> Self {
        Self {
            input,
            threshold,
            value,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the threshold operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            threshold: f32,
            value: f32,
        }

        let params = Params {
            size: size as u32,
            threshold: self.threshold,
            value: self.value,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Threshold Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Threshold Bind Group Layout"),
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
            label: Some("Threshold Bind Group"),
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
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Shader"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Threshold Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Threshold Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Threshold Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Threshold Pass"),
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

        device.submit_and_poll(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(
            output_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply threshold activation
    /// # Arguments
    /// * `threshold` - Threshold value
    /// * `value` - Replacement value for elements <= threshold
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn threshold_wgsl(self, threshold: f32, value: f32) -> Result<Self> {
        Threshold::new(self, threshold, value).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_threshold() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(data, vec![5], device.clone());

        let output = input.threshold_wgsl(0.0, -10.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], -10.0); // -2 <= 0, replaced
        assert_eq!(result[1], -10.0); // -1 <= 0, replaced
        assert_eq!(result[2], -10.0); // 0 <= 0, replaced
        assert_eq!(result[3], 1.0); // 1 > 0, unchanged
        assert_eq!(result[4], 2.0); // 2 > 0, unchanged
    }

    #[tokio::test]
    async fn test_threshold_custom() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.5, 1.0, 1.5, 2.0];
        let input = Tensor::new(data, vec![4], device.clone());

        let output = input.threshold_wgsl(1.0, 0.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.0); // 0.5 <= 1.0
        assert_eq!(result[1], 0.0); // 1.0 <= 1.0
        assert_eq!(result[2], 1.5); // 1.5 > 1.0
        assert_eq!(result[3], 2.0); // 2.0 > 1.0
    }
}
