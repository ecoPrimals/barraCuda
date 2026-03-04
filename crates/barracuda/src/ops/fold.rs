// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fold - Pure WGSL
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

/// Fold operation (col2im - inverse of unfold)
pub struct Fold {
    input: Tensor,
    output_size: (usize, usize),
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Fold {
    /// Create a new fold operation
    pub fn new(
        input: Tensor,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("Fold expects 3D tensor [B, C*K*K, L], got shape {shape:?}"),
            });
        }

        Ok(Self {
            input,
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/fold_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the fold operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels_times_kernel = shape[1];

        // Infer channels from input shape
        // channels_times_kernel = channels * kernel_height * kernel_width
        let kernel_elements = self.kernel_size.0 * self.kernel_size.1;
        let channels = channels_times_kernel / kernel_elements;

        if !channels_times_kernel.is_multiple_of(kernel_elements) {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Input channels*kernel ({channels_times_kernel}) must be divisible by kernel elements ({kernel_elements})"
                ),
            });
        }

        let out_height = self.output_size.0;
        let out_width = self.output_size.1;
        let output_size = batch_size * channels * out_height * out_width;

        // Compute number of blocks
        let num_blocks_h =
            ((out_height + 2 * self.padding - self.dilation * (self.kernel_size.0 - 1) - 1)
                / self.stride)
                + 1;
        let num_blocks_w =
            ((out_width + 2 * self.padding - self.dilation * (self.kernel_size.1 - 1) - 1)
                / self.stride)
                + 1;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            out_height: u32,
            out_width: u32,
            kernel_height: u32,
            kernel_width: u32,
            stride: u32,
            padding: u32,
            dilation: u32,
            num_blocks_h: u32,
            num_blocks_w: u32,
            _pad1: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_height: self.kernel_size.0 as u32,
            kernel_width: self.kernel_size.1 as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            dilation: self.dilation as u32,
            num_blocks_h: num_blocks_h as u32,
            num_blocks_w: num_blocks_w as u32,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fold Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Fold Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Fold Bind Group Layout"),
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
            label: Some("Fold Bind Group"),
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
                    label: Some("Fold Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Fold Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Fold Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fold Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);
            let workgroups_z = (batch_size * channels) as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
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
    async fn test_fold_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input shape: [B, C*K*K, L] where K=3, so C*9
        let data: Vec<f32> = (0..324).map(|i| i as f32).collect();
        let input = Tensor::from_data(
            &data,
            vec![1, 9, 36], // 1 channel * 9 kernel elements, 36 blocks
            device.clone(),
        )
        .unwrap();

        let folded = Fold::new(input, (6, 6), (3, 3), 1, 0, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(folded.shape(), &vec![1, 1, 6, 6]);
    }

    #[tokio::test]
    async fn test_fold_invalid_shape() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device.clone()).unwrap();

        assert!(Fold::new(input, (4, 4), (3, 3), 1, 0, 1).is_err());
    }

    #[tokio::test]
    async fn test_fold_with_padding() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..576).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 9, 64], device.clone()).unwrap();

        let folded = Fold::new(input, (8, 8), (3, 3), 1, 1, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(folded.shape(), &vec![1, 1, 8, 8]);
    }
}
