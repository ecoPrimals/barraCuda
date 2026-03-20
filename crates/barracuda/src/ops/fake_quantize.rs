// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fake quantization operation - Simulate quantization for training
//!
//! Fake quantization simulates the effect of quantization during training
//! by quantizing values to N bits but keeping them in floating point format.

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FakeQuantizeParams {
    size: u32,
    num_bits: u32,
    scale: f32,
    zero_point: f32,
}

/// Fake quantization operation
pub struct FakeQuantize {
    input: Tensor,
    num_bits: u32,
    scale: f32,
    zero_point: f32,
}

impl FakeQuantize {
    /// Create fake quantization operation
    /// # Errors
    /// Returns [`Err`] if `num_bits` is 0 or > 32, or scale is not positive.
    pub fn new(input: Tensor, num_bits: u32, scale: f32, zero_point: f32) -> Result<Self> {
        if num_bits == 0 || num_bits > 32 {
            return Err(BarracudaError::invalid_op(
                "fake_quantize",
                format!("num_bits must be between 1 and 32, got {num_bits}"),
            ));
        }
        if scale <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "fake_quantize",
                format!("scale must be positive, got {scale}"),
            ));
        }
        Ok(Self {
            input,
            num_bits,
            scale,
            zero_point,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/fake_quantize_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute fake quantization on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create params
        let params = FakeQuantizeParams {
            size: size as u32,
            num_bits: self.num_bits,
            scale: self.scale,
            zero_point: self.zero_point,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FakeQuantize Params"),
            size: std::mem::size_of::<FakeQuantizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FakeQuantize Bind Group Layout"),
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
            label: Some("FakeQuantize Bind Group"),
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

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("FakeQuantize"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FakeQuantize Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FakeQuantize Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("FakeQuantize Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FakeQuantize Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            use crate::device::{DeviceCapabilities, WorkloadType};
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_fake_quantize_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device)
            .await
            .unwrap();

        let output = FakeQuantize::new(input, 8, 1.0, 0.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Should quantize and dequantize values
        assert_eq!(result.len(), 5);
    }
}
