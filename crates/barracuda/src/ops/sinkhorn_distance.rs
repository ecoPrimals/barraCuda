// SPDX-License-Identifier: AGPL-3.0-or-later
//! SinkhornDistance - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Sinkhorn Distance operation
pub struct SinkhornDistance {
    dist1: Tensor,
    dist2: Tensor,
    cost_matrix: Tensor,
    num_iterations: u32,
    epsilon: f32,
}

impl SinkhornDistance {
    /// Create a new Sinkhorn distance operation
    pub fn new(
        dist1: Tensor,
        dist2: Tensor,
        cost_matrix: Tensor,
        num_iterations: Option<u32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        let dist1_size = dist1.shape().iter().product::<usize>();
        let dist2_size = dist2.shape().iter().product::<usize>();

        if dist1_size != dist2_size {
            return Err(BarracudaError::invalid_op(
                "sinkhorn_distance",
                "dist1 and dist2 must have same size",
            ));
        }

        let size = dist1_size;
        let cost_shape = cost_matrix.shape();
        if cost_shape.len() != 2 || cost_shape[0] != size || cost_shape[1] != size {
            return Err(BarracudaError::invalid_op(
                "sinkhorn_distance",
                "cost_matrix must be [size, size]",
            ));
        }

        Ok(Self {
            dist1,
            dist2,
            cost_matrix,
            num_iterations: num_iterations.unwrap_or(50),
            epsilon: epsilon.unwrap_or(0.1),
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/sinkhorn_distance_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the Sinkhorn distance operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.dist1.device();

        let size = self.dist1.shape().iter().product::<usize>();

        // Output is scalar distance
        let output_buffer = device.create_buffer_f32(1)?;

        // Transport plan buffer
        let transport_size = size * size;
        let transport_buffer = device.create_buffer_f32(transport_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            num_iterations: u32,
            epsilon: f32,
            _padding: u32,
        }

        let params = Params {
            size: size as u32,
            num_iterations: self.num_iterations,
            epsilon: self.epsilon,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SinkhornDistance Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("SinkhornDistance Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SinkhornDistance Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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
            label: Some("SinkhornDistance Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.dist1.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dist2.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.cost_matrix.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: transport_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SinkhornDistance Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SinkhornDistance Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SinkhornDistance Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SinkhornDistance Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_sinkhorn_distance_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 4;

        let dist1 = Tensor::from_vec_on(vec![0.25; size], vec![size], device.clone())
            .await
            .unwrap();

        let dist2 = Tensor::from_vec_on(vec![0.25; size], vec![size], device.clone())
            .await
            .unwrap();

        let cost_matrix =
            Tensor::from_vec_on(vec![1.0; size * size], vec![size, size], device.clone())
                .await
                .unwrap();

        let output = SinkhornDistance::new(dist1, dist2, cost_matrix, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[1]);
    }
}
