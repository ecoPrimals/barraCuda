// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Multiplication Operation
//!
//! **Operation**: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//! **Complexity**: O(1) - 4 multiplications + 2 additions
//! **Performance**: ~2-3 GPU cycles per operation
//!
//! **CRITICAL**: This operation is the bottleneck for FFT butterfly computations
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL (no unsafe)
//! - ✅ Hardware-agnostic
//! - ✅ Numerically precise (IEEE 754)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex multiplication operation
///
/// Multiplies two complex tensors element-wise.
/// This is the CRITICAL operation for FFT butterfly computations.
pub struct ComplexMul {
    input_a: Tensor,
    input_b: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexMul {
    /// Create a new complex multiplication operation
    pub fn new(input_a: Tensor, input_b: Tensor) -> Result<Self> {
        // Validate tensors
        if input_a.shape() != input_b.shape() {
            return Err(BarracudaError::Device(
                "Complex tensors must have same shape".to_string(),
            ));
        }

        let shape = input_a.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Complex tensors must have last dimension = 2 (real, imag)".to_string(),
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
                label: Some("Complex Mul Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("mul.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Complex Mul Bind Group Layout"),
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
                    label: Some("Complex Mul Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Complex Mul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
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

    /// Execute complex multiplication on GPU
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input_a.device();
        let num_elements = self.input_a.len();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Mul Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_elements as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Mul Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Mul Bind Group"),
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

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Complex Mul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Complex Mul Compute Pass"),
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
            self.input_a.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_mul_correctness() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // (3+4i) * (1+2i) = (3-8) + (6+4)i = -5+10i
        let data_a = vec![3.0f32, 4.0];
        let data_b = vec![1.0f32, 2.0];

        let tensor_a = Tensor::from_data(&data_a, vec![1, 2], device.clone()).unwrap();
        let tensor_b = Tensor::from_data(&data_b, vec![1, 2], device.clone()).unwrap();

        let op = ComplexMul::new(tensor_a, tensor_b).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - (-5.0)).abs() < 1e-5); // Real part
        assert!((result_data[1] - 10.0).abs() < 1e-5); // Imag part
    }

    #[tokio::test]
    async fn test_complex_mul_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // z * 1 = z
        let data_z = vec![3.0f32, 4.0];
        let data_one = vec![1.0f32, 0.0];

        let tensor_z = Tensor::from_data(&data_z, vec![1, 2], device.clone()).unwrap();
        let tensor_one = Tensor::from_data(&data_one, vec![1, 2], device.clone()).unwrap();

        let op = ComplexMul::new(tensor_z.clone(), tensor_one).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - 4.0).abs() < 1e-6);
    }
}
