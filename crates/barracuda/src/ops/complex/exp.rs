// SPDX-License-Identifier: AGPL-3.0-only
//! Complex Exponential Operation
//!
//! **Operation**: exp(a + bi) = exp(a)[cos(b) + i·sin(b)] (Euler's formula)
//! **Complexity**: O(1) - 1 exp + 2 trig functions
//! **CRITICAL**: This is THE operation for FFT twiddle factors `W_N^k` = exp(-2πik/N)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex exponential: exp(z) = exp(re)·[cos(im) + i·sin(im)] (Euler's formula).
pub struct ComplexExp {
    input: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexExp {
    /// Create complex exp operation. Input must have last dim = 2 (re, im).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor) -> Result<Self> {
        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Complex tensor must have last dimension = 2".to_string(),
            ));
        }

        let device = input.device();
        let shader = device.compile_shader(include_str!("exp.wgsl"), Some("Complex Exp Shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Complex Exp Bind Group Layout"),
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
                    label: Some("Complex Exp Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Complex Exp Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            input,
            pipeline,
            bind_group_layout,
        })
    }

    /// Execute complex exponential on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Exp Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_elements as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Exp Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Exp Bind Group"),
            layout: &self.bind_group_layout,
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
            label: Some("Complex Exp Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Complex Exp Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let num_complex = num_elements / 2;
            let workgroups = (num_complex as u32).div_ceil(optimal_wg_size);

            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

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

    #[tokio::test]
    async fn test_complex_exp_euler() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // exp(iπ) = cos(π) + i·sin(π) = -1 + 0i
        // (Euler's identity: exp(iπ) + 1 = 0)
        let pi = std::f32::consts::PI;
        let data = vec![0.0f32, pi]; // 0 + πi
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let op = ComplexExp::new(tensor).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - (-1.0)).abs() < 1e-5); // Real ≈ -1
        assert!((result_data[1] - 0.0).abs() < 1e-5); // Imag ≈ 0
    }

    #[tokio::test]
    async fn test_complex_exp_zero() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // exp(0) = 1+0i
        let data = vec![0.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();
        let result = ComplexExp::new(tensor)
            .unwrap()
            .execute()
            .unwrap()
            .to_vec()
            .unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }
}
