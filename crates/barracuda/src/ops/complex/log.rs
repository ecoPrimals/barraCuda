// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Logarithm

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex natural logarithm: log(z) = log|z| + i·arg(z).
pub struct ComplexLog {
    input: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexLog {
    /// Create complex log operation. Input must have last dim = 2 (re, im).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor) -> Result<Self> {
        if input.shape().last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Must have last dimension = 2".to_string(),
            ));
        }
        let device = input.device();
        let shader = device.compile_shader(include_str!("log.wgsl"), Some("Complex Log Shader"));
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("BGL"),
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
                    label: Some("PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Pipeline"),
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

    /// Execute complex logarithm on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.input.len();
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = [n as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("P"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG"),
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
        let mut encoder =
            device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: Some("E") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("P"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let wg =
                ((n / 2) as u32).div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));
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
    async fn test_complex_log_one() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test log(1+0i) = 0+0i
        let data = vec![1.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let log_op = ComplexLog::new(tensor).unwrap();
        let result = log_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 0.0).abs() < 1e-5,
            "log(1) real part should be 0"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "log(1) imag part should be 0"
        );
    }

    #[tokio::test]
    async fn test_complex_log_euler_base() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test log(e+0i) = 1+0i (approximately)
        let e = std::f32::consts::E;
        let data = vec![e, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let log_op = ComplexLog::new(tensor).unwrap();
        let result = log_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 1.0).abs() < 1e-5,
            "log(e) real part should be 1"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "log(e) imag part should be 0"
        );
    }
}
