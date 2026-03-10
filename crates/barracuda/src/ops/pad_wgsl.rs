// SPDX-License-Identifier: AGPL-3.0-only
//! Pad - Add padding to tensor - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its padding parameters
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Pad operation - Add padding to a 4D tensor (NCHW format)
pub struct Pad {
    input: Tensor,
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
    value: f32,
}

impl Pad {
    /// Create a new pad operation
    #[must_use]
    pub fn new(input: Tensor, padding: (usize, usize, usize, usize), value: f32) -> Self {
        Self {
            input,
            padding,
            value,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/pad_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the pad operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Assume NCHW format
        let batch_size = shape[0];
        let channels = shape[1];
        let input_height = shape[2];
        let input_width = shape[3];

        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;
        let output_height = input_height + pad_top + pad_bottom;
        let output_width = input_width + pad_left + pad_right;

        let output_size = batch_size * channels * output_height * output_width;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pad Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_width: u32,
            input_height: u32,
            output_width: u32,
            output_height: u32,
            channels: u32,
            batch_size: u32,
            pad_left: u32,
            pad_top: u32,
            pad_value: f32,
        }

        let params = Params {
            input_width: input_width as u32,
            input_height: input_height as u32,
            output_width: output_width as u32,
            output_height: output_height as u32,
            channels: channels as u32,
            batch_size: batch_size as u32,
            pad_left: pad_left as u32,
            pad_top: pad_top as u32,
            pad_value: self.value,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pad Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Pad Bind Group Layout"),
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
            label: Some("Pad Bind Group"),
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
                    label: Some("Pad Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Pad Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Pad Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pad Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (output_width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (output_height as u32).div_ceil(optimal_wg_size);
            let workgroups_z = (batch_size * channels) as u32;

            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![batch_size, channels, output_height, output_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Add padding to tensor
    ///
    /// # Arguments
    ///
    /// * `padding` - (left, right, top, bottom) padding amounts
    /// * `value` - Value to use for padding
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn pad_wgsl(self, padding: (usize, usize, usize, usize), value: f32) -> Result<Self> {
        Pad::new(self, padding, value).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_pad_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 1x1x2x2 tensor
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 2, 2], device.clone());

        // Pad by 1 on all sides
        let output = input.pad_wgsl((1, 1, 1, 1), 0.0).unwrap();

        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        let result = output.to_vec().unwrap();

        // Check padding (should be 0)
        assert_eq!(result[0], 0.0); // Top-left corner
        assert_eq!(result[3], 0.0); // Top-right corner

        // Check original data (should be preserved in center)
        assert_eq!(result[5], 1.0);
        assert_eq!(result[6], 2.0);
        assert_eq!(result[9], 3.0);
        assert_eq!(result[10], 4.0);
    }
}
