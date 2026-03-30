// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Absolute Value (Magnitude) Operation
//!
//! **Operation**: |a + bi| = sqrt(a² + b²)
//! **Complexity**: O(1) - native WGSL `length()` function
//! **Use Case**: Power spectra, structure factors S(q)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex absolute value (magnitude): |a + bi| = sqrt(a² + b²).
pub struct ComplexAbs {
    input: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ComplexAbs {
    /// Creates a new complex abs. Input last dimension must be 2 (real, imag).
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
        let shader = device.compile_shader(include_str!("abs.wgsl"), Some("Complex Abs Shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Complex Abs Bind Group Layout"),
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
                    label: Some("Complex Abs Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Complex Abs Pipeline"),
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

    /// Executes complex abs and returns real-valued magnitudes.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();
        let num_complex = num_elements / 2;

        // Output is real-valued (one f32 per complex number)
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Abs Output"),
            size: (num_complex * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Abs Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex Abs Bind Group"),
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
            label: Some("Complex Abs Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Complex Abs Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (num_complex as u32).div_ceil(optimal_wg_size);

            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Output shape is [batch..., 1] (real magnitudes)
        let mut output_shape = self.input.shape().to_vec();
        *output_shape
            .last_mut()
            .ok_or_else(|| BarracudaError::execution_failed("output_shape empty"))? = 1;

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_abs() {
        let device = crate::device::test_pool::get_test_device().await;

        // |3+4i| = 5
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let op = ComplexAbs::new(tensor).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 5.0).abs() < 1e-6);
    }
}
