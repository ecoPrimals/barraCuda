//! AdaptiveMaxPool1D - 1D Adaptive Max Pooling
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Applies max pooling with adaptive kernel size to produce fixed output size
//! Used in models like ResNet, VGG for variable input sizes

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaptiveMaxPool1DParams {
    batch_size: u32,
    channels: u32,
    in_length: u32,
    out_length: u32,
}

pub struct AdaptiveMaxPool1D {
    input: Tensor,
    output_length: usize,
}

impl AdaptiveMaxPool1D {
    pub fn new(input: Tensor, output_length: usize) -> Result<Self> {
        // Validate input shape: must be 3D [B, C, L]
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(BarracudaError::invalid_op(
                "adaptive_max_pool1d",
                "input must be 3D tensor [B, C, L]",
            ));
        }

        if output_length == 0 {
            return Err(BarracudaError::invalid_op(
                "adaptive_max_pool1d",
                "output_length must be positive",
            ));
        }

        Ok(Self {
            input,
            output_length,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/adaptive_max_pool1d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_length = shape[2];

        let output_size = batch_size * channels * self.output_length;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = AdaptiveMaxPool1DParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_length: in_length as u32,
            out_length: self.output_length as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adaptive_max_pool1d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("adaptive_max_pool1d_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("adaptive_max_pool1d_bind_group_layout"),
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
                    label: Some("adaptive_max_pool1d_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adaptive_max_pool1d_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adaptive_max_pool1d_bind_group"),
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

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adaptive_max_pool1d_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adaptive_max_pool1d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, self.output_length],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 1D adaptive max pooling to fixed output length
    pub fn adaptive_max_pool1d(self, output_length: usize) -> Result<Self> {
        AdaptiveMaxPool1D::new(self, output_length)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_adaptive_max_pool1d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0; 3 * 16];
        let input = Tensor::from_vec_on(input_data, vec![1, 3, 16], device.clone())
            .await
            .unwrap();

        let output = input.adaptive_max_pool1d(8).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 8]);
        assert_eq!(result.len(), 24);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_adaptive_max_pool1d_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Invalid shape (not 3D)
        let input = Tensor::from_vec_on(vec![1.0; 16], vec![4, 4], device.clone())
            .await
            .unwrap();
        assert!(input.adaptive_max_pool1d(8).is_err());
    }
}
