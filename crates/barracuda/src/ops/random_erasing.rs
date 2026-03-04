// SPDX-License-Identifier: AGPL-3.0-or-later
//! Random erasing data augmentation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Randomly erases rectangular regions in images

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomErasingParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    erase_value: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

pub struct RandomErasing {
    input: Tensor,
    erase_boxes: Tensor,
    erase_value: f32,
}

impl RandomErasing {
    /// Create RandomErasing operation
    pub fn new(input: Tensor, erase_boxes: Tensor, erase_value: f32) -> Result<Self> {
        // Validate erase_boxes shape: [batch_size, 4] (top, left, height, width)
        let erase_shape = erase_boxes.shape();
        if erase_shape.len() != 2 || erase_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomErasing",
                format!("erase_boxes must be 2D [batch_size, 4], got shape {erase_shape:?}"),
            ));
        }

        Ok(Self {
            input,
            erase_boxes,
            erase_value,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/random_erasing_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute RandomErasing on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomErasing",
                format!(
                    "input must be 4D [batch, channels, height, width], got shape {input_shape:?}"
                ),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        if self.erase_boxes.shape()[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "RandomErasing",
                format!(
                    "erase_boxes batch size {} must match input batch size {}",
                    self.erase_boxes.shape()[0],
                    batch_size
                ),
            ));
        }

        // Create output buffer: [batch, channels, height, width]
        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = RandomErasingParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            erase_value: self.erase_value,
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
                label: Some("RandomErasing Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RandomErasing Bind Group Layout"),
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
            label: Some("RandomErasing Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.erase_boxes.buffer().as_entire_binding(),
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
        let shader = device.compile_shader(Self::wgsl_shader(), Some("RandomErasing"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RandomErasing Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RandomErasing Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RandomErasing Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RandomErasing Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let (wg_x, wg_y, wg_z) = caps.optimal_workgroup_size_3d(WorkloadType::Convolution);
            let workgroups_x = (width as u32).div_ceil(wg_x);
            let workgroups_y = (height as u32).div_ceil(wg_y);
            let workgroups_z = ((batch_size * channels) as u32).div_ceil(wg_z);
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
    async fn test_random_erasing_basic() {
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

        let erase_boxes = Tensor::from_vec_on(
            vec![5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 8.0, 8.0], // [batch, 4] - (top, left, height, width)
            vec![batch_size, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = RandomErasing::new(input, erase_boxes, 0.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size, channels, height, width]);
    }
}
