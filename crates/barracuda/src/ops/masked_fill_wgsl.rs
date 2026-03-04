// SPDX-License-Identifier: AGPL-3.0-or-later
//! Masked Fill - Fill values where mask is true - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its fill value
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Masked Fill operation - Fill values where mask is true
///
/// Fills elements of the input tensor with value where mask is true (non-zero).
pub struct MaskedFill {
    input: Tensor,
    mask: Tensor,
    value: f32,
}

impl MaskedFill {
    /// Create a new MaskedFill operation
    pub fn new(input: Tensor, mask: Tensor, value: f32) -> Self {
        Self { input, mask, value }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/tensor/masked_fill_f64.wgsl"
                ))
            });
            &S
        }
    }

    /// Execute the masked fill operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Validate shapes match
        if self.mask.shape() != shape {
            return Err(BarracudaError::invalid_shape(
                shape.to_vec(),
                self.mask.shape().to_vec(),
            ));
        }

        let size: usize = shape.iter().product();

        // Convert mask to u32 (0 = false, 1 = true)
        let mask_data = self.mask.to_vec()?;
        let mask_u32: Vec<u32> = mask_data
            .iter()
            .map(|&v| if v != 0.0 { 1u32 } else { 0u32 })
            .collect();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let mask_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MaskedFill Mask"),
                contents: bytemuck::cast_slice(&mask_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            fill_value: f32,
        }

        let params = Params {
            size: size as u32,
            fill_value: self.value,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MaskedFill Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MaskedFill Bind Group Layout"),
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
            label: Some("MaskedFill Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask_buffer.as_entire_binding(),
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

        // Create compute pipeline
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Shader"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MaskedFill Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MaskedFill Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("MaskedFill Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MaskedFill Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Fill elements where mask is true with a value
    ///
    /// # Arguments
    ///
    /// * `mask` - Boolean mask tensor (non-zero = true)
    /// * `value` - Value to fill where mask is true
    ///
    /// # Returns
    ///
    /// Tensor with masked values filled
    pub fn masked_fill_wgsl(self, mask: Tensor, value: f32) -> Result<Self> {
        MaskedFill::new(self, mask, value).execute()
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
    async fn test_masked_fill_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: [1, 2, 3, 4]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![4], device.clone());

        // Mask: [0, 1, 0, 1] (fill positions 1 and 3)
        let mask_data = vec![0.0, 1.0, 0.0, 1.0];
        let mask = Tensor::new(mask_data, vec![4], device.clone());

        let output = input.masked_fill_wgsl(mask, 99.0).unwrap();

        assert_eq!(output.shape(), &[4]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 99.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 99.0);
    }

    #[tokio::test]
    async fn test_masked_fill_2d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Input: [[1,2], [3,4]]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![2, 2], device.clone());

        // Mask: [[1,0], [0,1]]
        let mask_data = vec![1.0, 0.0, 0.0, 1.0];
        let mask = Tensor::new(mask_data, vec![2, 2], device.clone());

        let output = input.masked_fill_wgsl(mask, -1.0).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], -1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], -1.0);
    }

    #[tokio::test]
    async fn test_masked_fill_all_false() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new(data.clone(), vec![3], device.clone());

        let mask_data = vec![0.0, 0.0, 0.0];
        let mask = Tensor::new(mask_data, vec![3], device.clone());

        let output = input.masked_fill_wgsl(mask, 99.0).unwrap();

        let result = output.to_vec().unwrap();
        // No changes - mask all false
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }
}
