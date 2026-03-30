// SPDX-License-Identifier: AGPL-3.0-or-later
//! Weight Normalization
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Reparameterizes weights as: w = g * (v / ||v||)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WeightNormParams {
    num_weights: u32,
    dim: u32,
    _padding: [u32; 2],
}

/// Weight normalization: reparameterizes weights as w = g * (v / ||v||).
pub struct WeightNorm {
    v: Tensor,
    g: Tensor,
    dim: u32,
}

impl WeightNorm {
    /// Create `WeightNorm` operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(v: Tensor, g: Tensor, dim: u32) -> Result<Self> {
        Ok(Self { v, g, dim })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/norm/weight_norm_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute `WeightNorm` on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.v.device();
        let v_shape = self.v.shape();

        if v_shape.is_empty() {
            return Err(BarracudaError::invalid_op(
                "WeightNorm",
                format!("v must have at least 1 dimension, got shape {v_shape:?}"),
            ));
        }

        let num_weights = self.v.len();

        // Create output buffer: same shape as v
        let output_buffer = device.create_buffer_f32(num_weights)?;

        let params = WeightNormParams {
            num_weights: num_weights as u32,
            dim: self.dim,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WeightNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("WeightNorm Bind Group Layout"),
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
            label: Some("WeightNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.g.buffer().as_entire_binding(),
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
        let shader = device.compile_shader(Self::wgsl_shader(), Some("WeightNorm"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("WeightNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("WeightNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("WeightNorm Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WeightNorm Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (num_weights as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            v_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_weight_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_weights = 20;

        let v = Tensor::from_vec_on(vec![1.0; num_weights], vec![num_weights], device.clone())
            .await
            .unwrap();

        let g = Tensor::from_vec_on(vec![2.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = WeightNorm::new(v, g, 0).unwrap().execute().unwrap();

        assert_eq!(result.shape(), &[num_weights]);
    }
}
