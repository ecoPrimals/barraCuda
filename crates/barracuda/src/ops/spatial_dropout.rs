// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SpatialDropout` - Spatial Dropout (Channel-wise dropout)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Drops entire feature maps (channels) instead of individual elements
//! More effective for convolutional networks
//!
//! Reference: "Efficient Object Localization Using Convolutional Networks" by Tompson et al.

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/dropout/spatial_dropout_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpatialDropoutParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    drop_prob: f32,
    training: u32,
    _padding: u32,
    _padding2: u32,
}

/// Channel-wise spatial dropout for convolutional networks.
pub struct SpatialDropout {
    input: Tensor,
    mask: Tensor,
    drop_prob: f32,
    training: bool,
}

impl SpatialDropout {
    /// Creates a new spatial dropout operation. Mask shape must be [B, C].
    /// # Errors
    /// Returns [`Err`] if input is not 4D, mask shape does not match [B, C], or `drop_prob` is not in [0, 1).
    pub fn new(input: Tensor, mask: Tensor, drop_prob: f32, training: bool) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        // Validate mask shape: must be [B, C]
        let mask_shape = mask.shape();
        if mask_shape.len() != 2
            || mask_shape[0] != input_shape[0]
            || mask_shape[1] != input_shape[1]
        {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "mask must be 2D tensor [B, C] matching input batch and channels",
            ));
        }

        if !(0.0..1.0).contains(&drop_prob) {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "drop_prob must be in range [0.0, 1.0)",
            ));
        }

        Ok(Self {
            input,
            mask,
            drop_prob,
            training,
        })
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Executes spatial dropout and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = SpatialDropoutParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            drop_prob: self.drop_prob,
            training: u32::from(self.training),
            _padding: 0,
            _padding2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spatial_dropout_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("spatial_dropout_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("spatial_dropout_bind_group_layout"),
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
                    label: Some("spatial_dropout_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spatial_dropout_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial_dropout_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.mask.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("spatial_dropout_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spatial_dropout_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups_x = (width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (height as u32).div_ceil(optimal_wg_size);
            let workgroups_z = batch_size * channels;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply spatial dropout (channel-wise dropout)
    /// # Arguments
    /// - `mask`: Channel mask tensor [B, C] (random values on CPU, passed in)
    /// - `drop_prob`: Dropout probability
    /// - `training`: Whether in training mode
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn spatial_dropout(self, mask: Tensor, drop_prob: f32, training: bool) -> Result<Self> {
        SpatialDropout::new(self, mask, drop_prob, training)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_spatial_dropout_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        let mask = Tensor::from_vec_on(vec![1.0; 3], vec![1, 3], device.clone())
            .await
            .unwrap();

        let output = input.spatial_dropout(mask, 0.5, true).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 4, 4]);
        assert_eq!(result.len(), 48);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
