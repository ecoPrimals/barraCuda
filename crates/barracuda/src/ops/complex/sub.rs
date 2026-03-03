//! Complex Subtraction Operation
//!
//! **Operation**: (a + bi) - (c + di) = (a-c) + (b-d)i
//! **Complexity**: O(1) - trivial (native vec2 subtraction)
//! **Performance**: 1 SIMD operation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

pub struct ComplexSub {
    input_a: Tensor,
    input_b: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexSub {
    pub fn new(input_a: Tensor, input_b: Tensor) -> Result<Self> {
        if input_a.shape() != input_b.shape() {
            return Err(BarracudaError::Device(
                "Complex tensors must have same shape".to_string(),
            ));
        }

        let shape = input_a.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Complex tensors must have last dimension = 2".to_string(),
            ));
        }

        if !std::ptr::eq(input_a.device().as_ref(), input_b.device().as_ref()) {
            return Err(BarracudaError::Device(
                "Tensors must be on the same device".to_string(),
            ));
        }

        let device = input_a.device();
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Complex Sub Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("sub.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Complex Sub Bind Group Layout"),
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
                    label: Some("Complex Sub Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Complex Sub Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            input_a,
            input_b,
            pipeline,
            bind_group_layout,
        })
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input_a.device();
        let num_elements = self.input_a.len();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Sub Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_elements as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Sub Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Sub Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.input_b.buffer().as_entire_binding(),
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

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Complex Sub Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Complex Sub Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let num_complex = num_elements / 2;
            let workgroups = (num_complex as u32).div_ceil(optimal_wg_size);

            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input_a.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_sub_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // (5+7i) - (2+3i) = 3+4i
        let data_a = vec![5.0f32, 7.0];
        let data_b = vec![2.0f32, 3.0];

        let tensor_a = Tensor::from_data(&data_a, vec![1, 2], device.clone()).unwrap();
        let tensor_b = Tensor::from_data(&data_b, vec![1, 2], device.clone()).unwrap();

        let op = ComplexSub::new(tensor_a, tensor_b).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - 4.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_euler_identity_via_add_sub() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // Test components for Euler's identity validation
        // conj(a+bi) + (a+bi) = 2a (imag cancels)
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device.clone()).unwrap();
        let op = ComplexSub::new(tensor.clone(), tensor.clone()).unwrap();
        let result = op.execute().unwrap().to_vec().unwrap();
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }
}
