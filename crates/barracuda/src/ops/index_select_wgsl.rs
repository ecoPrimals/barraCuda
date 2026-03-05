// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Select - Gather elements along a dimension - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its own dimension and indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Index Select operation - Gather elements along a dimension
///
/// Selects values from input tensor along a dimension using provided indices.
pub struct IndexSelect {
    input: Tensor,
    dim: usize,
    indices: Vec<usize>,
}

impl IndexSelect {
    /// Create a new IndexSelect operation
    pub fn new(input: Tensor, dim: usize, indices: Vec<usize>) -> Self {
        Self {
            input,
            dim,
            indices,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/index_select_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the index select operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Validate dimension
        if self.dim >= shape.len() {
            let mut expected_shape = shape.to_vec();
            expected_shape.push(self.dim); // Add dim as a hint
            return Err(BarracudaError::invalid_shape(
                expected_shape,
                shape.to_vec(),
            ));
        }

        // Calculate output shape
        let mut output_shape = shape.to_vec();
        output_shape[self.dim] = self.indices.len();

        // Calculate strides
        let dim_size = shape[self.dim];
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let total_size = outer_size * self.indices.len() * inner_size;

        // Convert indices to u32
        let indices_u32: Vec<u32> = self.indices.iter().map(|&i| i as u32).collect();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Indices"),
                contents: bytemuck::cast_slice(&indices_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndexSelect Output"),
            size: (total_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            total_size: u32,
            dim_size: u32,
            outer_size: u32,
            inner_size: u32,
            num_indices: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            total_size: total_size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
            num_indices: self.indices.len() as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("IndexSelect Bind Group Layout"),
                    entries: &[
                        // binding 0: uniform params (matches WGSL)
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
                        // binding 1: input storage (read)
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
                        // binding 2: indices storage (read)
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
                        // binding 3: output storage (read-write)
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
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IndexSelect Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Shader"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("IndexSelect Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("IndexSelect Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("IndexSelect Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IndexSelect Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (total_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, total_size)?;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Select elements along a dimension using indices
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to select from
    /// * `indices` - Indices to select
    ///
    /// # Returns
    ///
    /// Tensor with selected elements
    pub fn index_select_wgsl(self, dim: usize, indices: Vec<usize>) -> Result<Self> {
        IndexSelect::new(self, dim, indices).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_index_select_1d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: [5] = [0, 1, 2, 3, 4]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![5], device.clone());

        let output = input.index_select_wgsl(0, vec![1, 3, 4]).unwrap();

        assert_eq!(output.shape(), &[3]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
    }

    #[tokio::test]
    async fn test_index_select_2d_rows() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: [3, 2] = [[0,1], [2,3], [4,5]]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![3, 2], device.clone());

        let output = input.index_select_wgsl(0, vec![2, 0]).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        // Row 2: [4, 5], Row 0: [0, 1]
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
    }

    #[tokio::test]
    async fn test_index_select_2d_cols() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: [2, 3] = [[0,1,2], [3,4,5]]
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![2, 3], device.clone());

        let output = input.index_select_wgsl(1, vec![2, 0]).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        // Row 0: [2, 0], Row 1: [5, 3]
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 3.0);
    }
}
