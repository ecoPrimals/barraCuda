// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Square Root via polar form

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex square root via polar form: √z = √|z| · exp(i·arg(z)/2).
pub struct ComplexSqrt {
    input: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexSqrt {
    /// Create complex square root operation. Input must have last dim = 2 (re, im).
    /// # Errors
    /// Returns [`Err`] if input last dimension is not 2.
    pub fn new(input: Tensor) -> Result<Self> {
        if input.shape().last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Must have last dimension = 2".to_string(),
            ));
        }
        let device = input.device();
        let shader = device.compile_shader(include_str!("sqrt.wgsl"), Some("Complex Sqrt Shader"));
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Complex Sqrt BGL"),
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
                    label: Some("Complex Sqrt PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Complex Sqrt Pipeline"),
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

    /// Execute complex square root on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Sqrt Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = [num_elements as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Sqrt BG"),
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
            label: Some("Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let wg = ((num_elements / 2) as u32)
                .div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));
            pass.dispatch_workgroups(wg, 1, 1);
        }
        device.submit_commands(Some(encoder.finish()));
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
    async fn test_complex_sqrt_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test sqrt(4+0i) = 2+0i
        let data = vec![4.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let sqrt_op = ComplexSqrt::new(tensor).unwrap();
        let result = sqrt_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 2.0).abs() < 1e-5,
            "Real part should be ~2.0"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "Imag part should be ~0.0"
        );
    }

    #[tokio::test]
    async fn test_complex_sqrt_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test sqrt(z)^2 = z for z = 3+4i
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let sqrt_op = ComplexSqrt::new(tensor).unwrap();
        let sqrt_result = sqrt_op.execute().unwrap();

        // Square the result (re^2 - im^2, 2*re*im)
        let sqrt_data = sqrt_result.to_vec().unwrap();
        let re = sqrt_data[0];
        let im = sqrt_data[1];
        let squared_re = re * re - im * im;
        let squared_im = 2.0 * re * im;

        assert!((squared_re - 3.0).abs() < 1e-4, "Should recover real part");
        assert!((squared_im - 4.0).abs() < 1e-4, "Should recover imag part");
    }
}
