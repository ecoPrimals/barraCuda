// SPDX-License-Identifier: AGPL-3.0-or-later
//! PixelShuffle - Pixel Shuffle (Depth to Space)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Rearranges elements in a tensor from depth to spatial dimensions
//! Used in super-resolution networks (ESPCN, EDSR)
//!
//! Transform [B, C*r^2, H, W] → [B, C, H*r, W*r]

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PixelShuffleParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    upscale_factor: u32,
}

pub struct PixelShuffle {
    input: Tensor,
    upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(input: Tensor, upscale_factor: usize) -> Result<Self> {
        // Validate input shape: must be 4D [B, C*r^2, H, W]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "input must be 4D tensor [B, C*r^2, H, W]",
            ));
        }

        if upscale_factor == 0 {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "upscale_factor must be positive",
            ));
        }

        let in_channels = shape[1];
        if !in_channels.is_multiple_of(upscale_factor * upscale_factor) {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "input channels must be divisible by upscale_factor^2",
            ));
        }

        Ok(Self {
            input,
            upscale_factor,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/misc/pixel_shuffle_f64.wgsl"
                ))
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

        let out_channels = in_channels / (self.upscale_factor * self.upscale_factor);
        let out_height = in_height * self.upscale_factor;
        let out_width = in_width * self.upscale_factor;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = PixelShuffleParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            upscale_factor: self.upscale_factor as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pixel_shuffle_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("pixel_shuffle_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pixel_shuffle_bind_group_layout"),
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pixel_shuffle_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pixel_shuffle_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel_shuffle_bind_group"),
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
            label: Some("pixel_shuffle_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pixel_shuffle_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);
            let workgroups_z = batch_size * out_channels;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply pixel shuffle (depth to space)
    ///
    /// # Arguments
    /// - `upscale_factor`: Upscaling factor r (output will be H*r x W*r)
    pub fn pixel_shuffle(self, upscale_factor: usize) -> Result<Self> {
        PixelShuffle::new(self, upscale_factor)?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_pixel_shuffle_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // [B=1, C*r^2=4, H=2, W=2] with r=2 → [B=1, C=1, H=4, W=4]
        let input_data = vec![1.0; 4 * 2 * 2];
        let input = Tensor::from_vec_on(input_data, vec![1, 4, 2, 2], device.clone())
            .await
            .unwrap();

        let output = input.pixel_shuffle(2).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
