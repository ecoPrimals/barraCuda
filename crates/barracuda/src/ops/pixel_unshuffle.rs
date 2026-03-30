// SPDX-License-Identifier: AGPL-3.0-or-later
//! PixelUnshuffle - Pixel Unshuffle (Space to Depth)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Rearranges elements in a tensor from spatial dimensions to depth
//! Inverse of pixel shuffle
//!
//! Transform [B, C, H, W] → [B, C*r^2, H/r, W/r]

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PixelUnshuffleParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    downscale_factor: u32,
}

pub struct PixelUnshuffle {
    input: Tensor,
    downscale_factor: usize,
}

impl PixelUnshuffle {
    pub fn new(input: Tensor, downscale_factor: usize) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "pixel_unshuffle",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if downscale_factor == 0 {
            return Err(BarracudaError::invalid_op(
                "pixel_unshuffle",
                "downscale_factor must be positive",
            ));
        }

        let height = shape[2];
        let width = shape[3];

        if height % downscale_factor != 0 || width % downscale_factor != 0 {
            return Err(BarracudaError::invalid_op(
                "pixel_unshuffle",
                "height and width must be divisible by downscale_factor",
            ));
        }

        Ok(Self {
            input,
            downscale_factor,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!(
                    "../shaders/misc/pixel_unshuffle_f64.wgsl"
                ).to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let in_channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let out_channels = in_channels * self.downscale_factor * self.downscale_factor;
        let out_height = in_height / self.downscale_factor;
        let out_width = in_width / self.downscale_factor;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = PixelUnshuffleParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            downscale_factor: self.downscale_factor as u32,
        };

        let params_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel_unshuffle_params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("pixel_unshuffle_shader"));

        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("pixel_unshuffle_bind_group_layout"),
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
            },
        );

        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pixel_unshuffle_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
        });

        let pipeline = device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pixel_unshuffle_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel_unshuffle_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
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

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("pixel_unshuffle_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pixel_unshuffle_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(&device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32 + optimal_wg_size - 1) / optimal_wg_size;
            let workgroups_y = (out_height as u32 + optimal_wg_size - 1) / optimal_wg_size;
            let workgroups_z = batch_size * out_channels;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply pixel unshuffle (space to depth)
    ///
    /// # Arguments
    /// - `downscale_factor`: Downscaling factor r (output will be H/r x W/r)
    pub fn pixel_unshuffle(self, downscale_factor: usize) -> Result<Self> {
        PixelUnshuffle::new(self, downscale_factor)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device;

    #[tokio::test]
    async fn test_pixel_unshuffle_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // [B=1, C=1, H=4, W=4] with r=2 → [B=1, C=4, H=2, W=2]
        let input_data = vec![1.0; 1 * 1 * 4 * 4];
        let input = Tensor::from_vec_on(input_data, vec![1, 1, 4, 4], device.clone())
            .await
            .unwrap();

        let output = input.pixel_unshuffle(2).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 4, 2, 2]);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
