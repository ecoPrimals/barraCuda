// SPDX-License-Identifier: AGPL-3.0-or-later
//! Random rotation augmentation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Rotates images by random angles
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/augmentation/random_rotation_f64.wgsl");

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomRotationParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    fill_value: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

pub struct RandomRotation {
    input: Tensor,
    rotation_matrices: Tensor,
    fill_value: f32,
}

impl RandomRotation {
    /// Create RandomRotation operation
    pub fn new(input: Tensor, rotation_matrices: Tensor, fill_value: f32) -> Result<Self> {
        // Validate rotation_matrices shape: [batch_size, 4] (cos, -sin, sin, cos)
        let rot_shape = rotation_matrices.shape();
        if rot_shape.len() != 2 || rot_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!("rotation_matrices must be 2D [batch_size, 4], got shape {rot_shape:?}"),
            ));
        }

        Ok(Self {
            input,
            rotation_matrices,
            fill_value,
        })
    }

    /// WGSL shader source (f64 canonical, downcast to f32 at compile)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
        });
        SHADER.as_str()
    }

    /// Execute RandomRotation on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!(
                    "input must be 4D [batch, channels, height, width], got shape {input_shape:?}"
                ),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        if self.rotation_matrices.shape()[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!(
                    "rotation_matrices batch size {} must match input batch size {}",
                    self.rotation_matrices.shape()[0],
                    batch_size
                ),
            ));
        }

        // Create output buffer: [batch, channels, height, width]
        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = RandomRotationParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            fill_value: self.fill_value,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomRotation Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RandomRotation Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RandomRotation Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rotation_matrices.buffer().as_entire_binding(),
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

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("RandomRotation"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RandomRotation Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RandomRotation Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RandomRotation Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RandomRotation Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (8x8x1 threads per workgroup)
            let workgroups_x = (width as u32).div_ceil(8);
            let workgroups_y = (height as u32).div_ceil(8);
            let workgroups_z = (batch_size * channels) as u32;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_random_rotation_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * channels * height * width],
            vec![batch_size, channels, height, width],
            device.clone(),
        )
        .await
        .unwrap();

        // Rotation matrices: [cos, -sin, sin, cos] for each batch item
        let rotation_matrices = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 1.0, 0.707, -0.707, 0.707, 0.707], // Identity and 45° rotation
            vec![batch_size, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = RandomRotation::new(input, rotation_matrices, 0.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size, channels, height, width]);
    }
}
