// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Select - Select elements by indices - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Index Select operation - Select elements by indices
pub struct IndexSelect {
    input: Tensor,
    indices: Vec<u32>,
}

impl IndexSelect {
    /// Create a new index select operation
    /// # Errors
    /// Returns [`Err`] if any index is out of bounds for the input size.
    pub fn new(input: Tensor, indices: Vec<u32>) -> Result<Self> {
        let input_size = input.shape().iter().product::<usize>();

        // Validate indices are in bounds
        for &idx in &indices {
            if idx as usize >= input_size {
                return Err(BarracudaError::invalid_op(
                    "IndexSelect",
                    format!("Index {idx} out of bounds for input size {input_size}"),
                ));
            }
        }

        Ok(Self { input, indices })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/index_select_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the index select operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails
    /// (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let num_indices = self.indices.len();

        // Output shape: replace first dimension with number of indices
        let mut output_shape = input_shape.to_vec();
        if output_shape.is_empty() {
            output_shape.push(num_indices);
        } else {
            output_shape[0] = num_indices;
        }

        let output_size = output_shape.iter().product::<usize>();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("IndexSelect Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndexSelect Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters - must match WGSL Params struct (32 bytes)
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

        let input_size = input_shape.iter().product::<usize>() as u32;
        let params = Params {
            total_size: output_size as u32,
            dim_size: input_size,
            outer_size: 1,
            inner_size: 1,
            num_indices: num_indices as u32,
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

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("IndexSelect Shader"));

        // Create compute pipeline
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
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

impl Tensor {
    /// Select elements by indices
    /// # Arguments
    /// * `indices` - Indices to select
    /// # Errors
    /// Returns [`Err`] if any index is out of bounds, buffer allocation fails, GPU dispatch fails,
    /// or buffer readback fails (e.g. device lost).
    pub fn index_select(self, indices: Vec<u32>) -> Result<Self> {
        IndexSelect::new(self, indices)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_select_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device);
        let result = input.index_select(vec![0, 2, 4]).unwrap();

        assert_eq!(result.shape(), &[3]);
        let output_data = result.to_vec().unwrap();
        assert_eq!(output_data[0], 1.0);
        assert_eq!(output_data[1], 3.0);
        assert_eq!(output_data[2], 5.0);
    }
}
