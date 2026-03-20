// SPDX-License-Identifier: AGPL-3.0-or-later
//! Group Normalization - Normalize within groups - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its group parameters
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Group normalization operation
pub struct GroupNorm {
    input: Tensor,
    num_groups: usize,
    epsilon: f32,
}

impl GroupNorm {
    /// Create a new group normalization operation
    #[must_use]
    pub fn new(input: Tensor, num_groups: usize, epsilon: f32) -> Self {
        Self {
            input,
            num_groups,
            epsilon,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/norm/group_norm_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the group normalization operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();

        // Assume NCHW format: [batch, channels, height, width]
        let batch = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();
        let group_size = channels / self.num_groups;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch: u32,
            channels: u32,
            num_groups: u32,
            group_size: u32,
            spatial_size: u32,
            epsilon: f32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            num_groups: self.num_groups as u32,
            group_size: group_size as u32,
            spatial_size: spatial_size as u32,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GroupNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GroupNorm Bind Group Layout"),
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
            label: Some("GroupNorm Bind Group"),
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
                    label: Some("GroupNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GroupNorm Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GroupNorm Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GroupNorm Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch (vendor-optimized workgroups)
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let total_groups = (batch * self.num_groups) as u32;
            let workgroups = total_groups.div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Apply group normalization (NCHW format)
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to divide channels into
    /// * `epsilon` - Small constant for numerical stability (default: 1e-5)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn group_norm_wgsl(self, num_groups: usize, epsilon: f32) -> Result<Self> {
        GroupNorm::new(self, num_groups, epsilon).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_group_norm_simple() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 1 batch, 4 channels, 1x1 spatial, 2 groups
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 4, 1, 1], device);

        let output = input.group_norm_wgsl(2, 1e-5).unwrap();

        assert_eq!(output.shape(), &[1, 4, 1, 1]);

        // Each group should be normalized independently
        let result = output.to_vec().unwrap();
        // Group 1: [1.0, 2.0] -> mean ~0, std ~1
        // Group 2: [3.0, 4.0] -> mean ~0, std ~1
        assert_eq!(result.len(), 4);
    }

    #[tokio::test]
    async fn test_group_norm_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 2 batches, 2 channels, 1x1 spatial, 1 group
        let data = vec![
            1.0, 2.0, // batch 0
            3.0, 4.0, // batch 1
        ];
        let input = Tensor::new(data, vec![2, 2, 1, 1], device);

        let output = input.group_norm_wgsl(1, 1e-5).unwrap();

        assert_eq!(output.shape(), &[2, 2, 1, 1]);
        assert_eq!(output.to_vec().unwrap().len(), 4);
    }
}
