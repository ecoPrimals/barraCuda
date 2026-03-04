// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scatter ND - N-dimensional scatter operation - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its multi-dimensional indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Scatter ND operation - N-dimensional scatter
pub struct ScatterNd {
    input: Tensor,
    indices: Tensor, // Shape: [batch_size, num_indices, index_rank]
    values: Tensor,
}

impl ScatterNd {
    /// Create a new scatter ND operation
    pub fn new(input: Tensor, indices: Tensor, values: Tensor) -> Result<Self> {
        let input_shape = input.shape();
        let indices_shape = indices.shape();
        let values_shape = values.shape();

        // Validate indices shape: should be [batch_size, num_indices, index_rank]
        if indices_shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "ScatterNd",
                format!(
                    "Indices must have at least 2 dimensions, got {}",
                    indices_shape.len()
                ),
            ));
        }

        let index_rank = indices_shape[indices_shape.len() - 1];
        if index_rank > input_shape.len() {
            return Err(BarracudaError::invalid_op(
                "ScatterNd",
                format!(
                    "Index rank {} exceeds input rank {}",
                    index_rank,
                    input_shape.len()
                ),
            ));
        }

        // Validate values shape matches expected output shape
        let num_indices = if indices_shape.len() > 1 {
            indices_shape[indices_shape.len() - 2]
        } else {
            1
        };
        let batch_size = if indices_shape.len() > 2 {
            indices_shape[..indices_shape.len() - 2].iter().product()
        } else {
            1
        };

        let mut expected_values_shape = vec![batch_size, num_indices];
        if index_rank < input_shape.len() {
            expected_values_shape.extend_from_slice(&input_shape[index_rank..]);
        }

        if values_shape != expected_values_shape {
            return Err(BarracudaError::invalid_op(
                "ScatterNd",
                format!(
                    "Values shape {values_shape:?} doesn't match expected shape {expected_values_shape:?}"
                ),
            ));
        }

        Ok(Self {
            input,
            indices,
            values,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/scatter_nd_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the scatter ND operation (modifies input in-place)
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let indices_shape = self.indices.shape();
        let values_shape = self.values.shape();
        let input_size: usize = input_shape.iter().product();
        let indices_size: usize = indices_shape.iter().product();
        let values_size: usize = values_shape.iter().product();

        // Access buffers directly (zero-copy)
        let input_buffer = self.input.buffer();
        let indices_buffer = self.indices.buffer();
        let values_buffer = self.values.buffer();

        // Create buffers for shape information
        let input_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ScatterNd Input Shape"),
                    contents: bytemuck::cast_slice(
                        &input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let indices_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ScatterNd Indices Shape"),
                    contents: bytemuck::cast_slice(
                        &indices_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let values_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ScatterNd Values Shape"),
                    contents: bytemuck::cast_slice(
                        &values_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_size: u32,
            indices_size: u32,
            values_size: u32,
            input_rank: u32,
            indices_rank: u32,
            index_rank: u32,
            batch_size: u32,
            num_indices: u32,
        }

        let index_rank = indices_shape[indices_shape.len() - 1];
        let num_indices = if indices_shape.len() > 1 {
            indices_shape[indices_shape.len() - 2]
        } else {
            1
        };
        let batch_size = if indices_shape.len() > 2 {
            indices_shape[..indices_shape.len() - 2].iter().product()
        } else {
            1
        };

        let params = Params {
            input_size: input_size as u32,
            indices_size: indices_size as u32,
            values_size: values_size as u32,
            input_rank: input_shape.len() as u32,
            indices_rank: indices_shape.len() as u32,
            index_rank: index_rank as u32,
            batch_size: batch_size as u32,
            num_indices: num_indices as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ScatterNd Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ScatterNd Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ScatterNd Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input_shape_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: indices_shape_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: values_shape_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("ScatterNd Shader"));

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ScatterNd Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("ScatterNd Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("ScatterNd Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ScatterNd Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(values_size as u32);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let output_data = crate::utils::read_buffer(device, input_buffer, input_size)?;
        Ok(Tensor::new(
            output_data,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Scatter values using N-dimensional indices (modifies input in-place)
    ///
    /// # Arguments
    ///
    /// * `indices` - N-dimensional indices tensor
    /// * `values` - Values to scatter
    pub fn scatter_nd(self, indices: Tensor, values: Tensor) -> Result<Self> {
        ScatterNd::new(self, indices, values)?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_scatter_nd_2d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: 3x3 matrix initialized to zeros
        let input = Tensor::new(vec![0.0; 9], vec![3, 3], device.clone());

        // Indices: [[0,0], [2,1]]
        let indices_data: Vec<f32> = vec![0.0, 0.0, 2.0, 1.0];
        let indices = Tensor::new(indices_data, vec![2, 2], device.clone());

        // Values: shape [1, 2] to match expected (batch_size=1, num_indices=2)
        let values = Tensor::new(vec![10.0, 20.0], vec![1, 2], device.clone());

        let result = input.scatter_nd(indices, values).unwrap();
        let output_data = result.to_vec().unwrap();

        // Expected: input[0,0] = 10.0, input[2,1] = 20.0
        assert_eq!(output_data[0], 10.0); // [0,0]
        assert_eq!(output_data[7], 20.0); // [2,1]
    }
}
