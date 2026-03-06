// SPDX-License-Identifier: AGPL-3.0-or-later
//! `LocalResponseNorm` - Local Response Normalization (LRN)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Normalizes activations within local neighborhoods
//! Used in `AlexNet` and other early CNNs
//!
//! Formula: `y_i` = `x_i` / (k + alpha * `sum(x_j^2)` / size)^beta

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalResponseNormParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    size: u32,
    alpha: f32,
    beta: f32,
    k: f32,
}

/// Local response normalization (AlexNet-style LRN).
pub struct LocalResponseNorm {
    input: Tensor,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
}

impl LocalResponseNorm {
    /// Creates a new LRN. Size is the normalization window; alpha, beta, k are formula parameters.
    /// # Errors
    /// Returns [`Err`] if input is not 4D [B, C, H, W], or if size is zero.
    pub fn new(input: Tensor, size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "local_response_norm",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if size == 0 {
            return Err(BarracudaError::invalid_op(
                "local_response_norm",
                "size must be positive",
            ));
        }

        Ok(Self {
            input,
            size,
            alpha,
            beta,
            k,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/norm/local_response_norm_f64.wgsl"
            ))
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes LRN and returns the normalized tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = LocalResponseNormParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            size: self.size as u32,
            alpha: self.alpha,
            beta: self.beta,
            k: self.k,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("local_response_norm_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("local_response_norm_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("local_response_norm_bind_group_layout"),
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
                    label: Some("local_response_norm_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("local_response_norm_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("local_response_norm_bind_group"),
            layout: &bind_group_layout,
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
            label: Some("local_response_norm_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("local_response_norm_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups_x = (width as u32).div_ceil(8);
            let workgroups_y = (height as u32).div_ceil(8);
            let workgroups_z = ((batch_size * channels) as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply local response normalization
    /// # Arguments
    /// - `size`: Neighborhood size
    /// - `alpha`: Scaling parameter (typically 1e-4)
    /// - `beta`: Exponent (typically 0.75)
    /// - `k`: Bias (typically 1.0 or 2.0)
    /// # Errors
    /// Returns [`Err`] if input is not 4D, size is zero, buffer allocation fails, GPU dispatch
    /// fails, or the device is lost.
    pub fn local_response_norm(self, size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self> {
        LocalResponseNorm::new(self, size, alpha, beta, k)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_local_response_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();

        let output = input.local_response_norm(5, 1e-4, 0.75, 1.0).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 4, 4]);
        assert_eq!(result.len(), 48);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
