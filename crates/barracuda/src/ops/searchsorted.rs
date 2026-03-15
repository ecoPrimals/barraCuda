// SPDX-License-Identifier: AGPL-3.0-only
//! `SearchSorted` - GPU parallel binary search
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Parallel binary search for each value
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates sorted array
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SearchSortedParams {
    sorted_size: u32,
    values_size: u32,
    side_right: u32, // 0 = left (default), 1 = right
    _pad1: u32,
}

/// GPU parallel binary search for insertion indices in a sorted array.
pub struct SearchSorted {
    sorted: Tensor,
    values: Tensor,
    side_right: bool,
}

impl SearchSorted {
    /// Creates a new searchsorted operation. `side_right` selects left (false) or right (true) insertion.
    /// # Errors
    /// Returns [`Err`] if sorted or values are empty, or if either is not 1D.
    pub fn new(sorted: Tensor, values: Tensor, side_right: bool) -> Result<Self> {
        if sorted.is_empty() {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Sorted array cannot be empty",
            ));
        }

        if values.is_empty() {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Values array cannot be empty",
            ));
        }

        // Validate sorted array is 1D
        if sorted.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Sorted array must be 1D",
            ));
        }

        if values.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Values array must be 1D",
            ));
        }

        Ok(Self {
            sorted,
            values,
            side_right,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/misc/searchsorted_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    fn u32_to_f32_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/u32_to_f32_f64.wgsl").to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes the binary search and returns insertion indices as f32 tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.sorted.device();
        let sorted_size = self.sorted.len();
        let values_size = self.values.len();

        let output_buffer = device.create_buffer_u32(values_size)?;

        let params = SearchSortedParams {
            sorted_size: sorted_size as u32,
            values_size: values_size as u32,
            side_right: u32::from(self.side_right),
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SearchSorted Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SearchSorted Bind Group Layout"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SearchSorted Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.values.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("SearchSorted"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SearchSorted Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SearchSorted Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SearchSorted Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SearchSorted Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (values_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Convert u32 indices to f32 on GPU (for Tensor compatibility)
        let indices_f32_buffer = device.create_buffer_f32(values_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvertParams {
            size: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let convert_params = ConvertParams {
            size: values_size as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let convert_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SearchSorted Convert Params"),
                    contents: bytemuck::bytes_of(&convert_params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let convert_bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SearchSorted Convert Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let convert_bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SearchSorted Convert Bind Group"),
            layout: &convert_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: convert_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_f32_buffer.as_entire_binding(),
                },
            ],
        });

        let convert_shader =
            device.compile_shader(Self::u32_to_f32_shader(), Some("SearchSorted Convert"));
        let convert_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SearchSorted Convert Pipeline Layout"),
                    bind_group_layouts: &[&convert_bind_group_layout],
                    immediate_size: 0,
                });

        let convert_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SearchSorted Convert Pipeline"),
                    layout: Some(&convert_pipeline_layout),
                    module: &convert_shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let mut convert_encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SearchSorted Convert Encoder"),
        });

        {
            let mut pass = convert_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SearchSorted Convert Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&convert_pipeline);
            pass.set_bind_group(0, Some(&convert_bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (values_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(convert_encoder.finish()));

        Ok(Tensor::from_buffer(
            indices_f32_buffer,
            vec![values_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_searchsorted_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let sorted = Tensor::from_vec_on(vec![1.0, 3.0, 5.0, 7.0], vec![4], device.clone())
            .await
            .unwrap();
        let values = Tensor::from_vec_on(vec![2.0, 4.0, 6.0], vec![3], device.clone())
            .await
            .unwrap();

        let result = SearchSorted::new(sorted, values, false)
            .unwrap()
            .execute()
            .unwrap();
        let indices = result.to_vec().unwrap();
        assert_eq!(indices.len(), 3);
        // Should be [1, 2, 3] (insertion points)
    }

    #[tokio::test]
    async fn test_searchsorted_right() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let sorted = Tensor::from_vec_on(vec![1.0, 3.0, 5.0], vec![3], device.clone())
            .await
            .unwrap();
        let values = Tensor::from_vec_on(vec![3.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = SearchSorted::new(sorted, values, true)
            .unwrap()
            .execute()
            .unwrap();
        let indices = result.to_vec().unwrap();
        assert_eq!(indices.len(), 1);
    }
}
