// SPDX-License-Identifier: AGPL-3.0-or-later
//! Upsample - Pure WGSL
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

/// Upsample operation
pub struct Upsample {
    input: Tensor,
    size: Option<(usize, usize)>,
    scale_factor: Option<(f32, f32)>,
    mode: UpsampleMode,
    align_corners: bool,
}

/// Upsampling interpolation mode.
#[derive(Debug, Clone, Copy)]
pub enum UpsampleMode {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
}

impl Upsample {
    /// Create a new upsample operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if input is not 4D [B, C, H, W], or neither size nor `scale_factor` is provided.
    pub fn new(
        input: Tensor,
        size: Option<(usize, usize)>,
        scale_factor: Option<(f32, f32)>,
        mode: UpsampleMode,
        align_corners: bool,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("Upsample expects 4D tensor [B, C, H, W], got shape {shape:?}"),
            });
        }

        if size.is_none() && scale_factor.is_none() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "Either size or scale_factor must be provided".to_string(),
            });
        }

        Ok(Self {
            input,
            size,
            scale_factor,
            mode,
            align_corners,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/misc/upsample_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the upsample operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        // Compute output size
        let (out_height, out_width) = if let Some((h, w)) = self.size {
            (h, w)
        } else if let Some((sh, sw)) = self.scale_factor {
            (
                (in_height as f32 * sh) as usize,
                (in_width as f32 * sw) as usize,
            )
        } else {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "Either size or scale_factor must be provided".to_string(),
            });
        };

        let output_size = batch_size * channels * out_height * out_width;

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
            in_height: u32,
            in_width: u32,
            out_height: u32,
            out_width: u32,
            mode: u32,
            align_corners: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            mode: match self.mode {
                UpsampleMode::Nearest => 0,
                UpsampleMode::Bilinear => 1,
            },
            align_corners: u32::from(self.align_corners),
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Upsample Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Upsample Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Upsample Bind Group Layout"),
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
            label: Some("Upsample Bind Group"),
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
                    label: Some("Upsample Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Upsample Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Upsample Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Upsample Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let (wg_x, wg_y) = caps.optimal_workgroup_size_2d(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32).div_ceil(wg_x);
            let workgroups_y = (out_height as u32).div_ceil(wg_y);
            let workgroups_z = (batch_size * channels) as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_commands(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsample_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 3, 4], device).unwrap();

        let upsampled = Upsample::new(input, Some((6, 8)), None, UpsampleMode::Nearest, false)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 6, 8]);
    }

    #[tokio::test]
    async fn test_upsample_scale_factor() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 2, 4], device).unwrap();

        let upsampled = Upsample::new(input, None, Some((2.0, 2.0)), UpsampleMode::Bilinear, false)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 4, 8]);
    }

    #[tokio::test]
    async fn test_upsample_invalid_shape() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Upsample::new(input, Some((10, 10)), None, UpsampleMode::Nearest, false,).is_err());
    }

    #[tokio::test]
    async fn test_upsample_no_params() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0; 12], vec![1, 1, 3, 4], device).unwrap();

        assert!(Upsample::new(input, None, None, UpsampleMode::Nearest, false,).is_err());
    }

    #[tokio::test]
    async fn test_upsample_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 16, 16], device).unwrap();

        let upsampled = Upsample::new(input, Some((32, 32)), None, UpsampleMode::Bilinear, true)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 32, 32]);
    }
}
