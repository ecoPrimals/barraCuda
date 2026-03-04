// SPDX-License-Identifier: AGPL-3.0-or-later
//! LayerScale - Per-layer learnable scaling
//!
//! **Canonical BarraCuda Pattern**: Struct with new/execute
//!
//! Used in vision transformers (CaiT, LeViT) to stabilize training.
//!
//! ## Algorithm
//!
//! ```text
//! LayerScale(x) = gamma ⊙ x
//! ```
//!
//! Where gamma is a learnable per-channel parameter.

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// LayerScale operation
pub struct LayerScale {
    input: Tensor,
    gamma: Tensor,
}

impl LayerScale {
    /// Create a new layer scale operation
    pub fn new(input: Tensor, gamma: Tensor) -> Result<Self> {
        // Validate shapes match
        if input.shape() != gamma.shape() {
            return Err(BarracudaError::shape_mismatch(
                input.shape().to_vec(),
                gamma.shape().to_vec(),
            ));
        }

        Ok(Self { input, gamma })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/layer_scale_f64.wgsl"
            ))
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Execute the layer scale operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            _padding: [u32; 3],
        }

        let params = Params {
            size: size as u32,
            _padding: [0; 3],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LayerScale Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("LayerScale Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LayerScale Bind Group Layout"),
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
            label: Some("LayerScale Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.gamma.buffer().as_entire_binding(),
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
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LayerScale Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("LayerScale Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("LayerScale Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerScale Pass"),
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

        // Return tensor with same shape as input
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_layer_scale_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.1, 0.2, 0.3], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.4).abs() < 1e-5);
        assert!((result[2] - 0.9).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_layer_scale_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Single element
        let input = Tensor::from_vec_on(vec![5.0], vec![1], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.5], vec![1], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5).abs() < 1e-5);

        // All zeros
        let input = Tensor::from_vec_on(vec![0.0, 0.0, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_layer_scale_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Gamma = 0 (complete suppression)
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.0, 0.0, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-5));

        // Gamma = 1 (identity)
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![1.0, 1.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input.clone(), gamma)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();
        let input_vec = input.to_vec().unwrap();
        for (r, i) in result.iter().zip(input_vec.iter()) {
            assert!((r - i).abs() < 1e-5);
        }
    }
}
