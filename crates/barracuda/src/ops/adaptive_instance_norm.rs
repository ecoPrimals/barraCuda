// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adaptive Instance Normalization (`AdaIN`) - Style transfer
//!
//! Transfers style from one image to another.
//! Used in neural style transfer, GANs.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/norm/adaptive_instance_norm_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// `AdaptiveInstanceNorm` operation
pub struct AdaptiveInstanceNorm {
    content: Tensor,
    style_mean: Tensor,
    style_std: Tensor,
}

impl AdaptiveInstanceNorm {
    /// Create a new adaptive instance norm operation
    /// # Errors
    /// Returns [`Err`] if content is not 4D (NCHW), style mean/std are not 1D, or style statistics
    /// do not match the number of channels.
    pub fn new(content: Tensor, style_mean: Tensor, style_std: Tensor) -> Result<Self> {
        let content_shape = content.shape();
        let style_mean_shape = style_mean.shape();
        let style_std_shape = style_std.shape();

        if content_shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_op(
                "AdaptiveInstanceNorm",
                format!("Content must be 4D (NCHW), got {}D", content_shape.len()),
            ));
        }

        if style_mean_shape.len() != 1 || style_std_shape.len() != 1 {
            return Err(crate::error::BarracudaError::invalid_op(
                "AdaptiveInstanceNorm",
                "Style mean and std must be 1D tensors",
            ));
        }

        if style_mean_shape[0] != content_shape[1] || style_std_shape[0] != content_shape[1] {
            return Err(crate::error::BarracudaError::invalid_op(
                "AdaptiveInstanceNorm",
                "Style statistics must match number of channels",
            ));
        }

        Ok(Self {
            content,
            style_mean,
            style_std,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the adaptive instance norm operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails,
    /// or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.content.device();
        let shape = self.content.shape();

        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial_size = height * width;
        let output_size = batch * channels * spatial_size;

        // Create buffers
        let content_buffer = self.content.buffer();
        let style_mean_buffer = self.style_mean.buffer();
        let style_std_buffer = self.style_std.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AdaptiveInstanceNorm Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch: u32,
            channels: u32,
            height: u32,
            width: u32,
            spatial_size: u32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            spatial_size: spatial_size as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("AdaptiveInstanceNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("AdaptiveInstanceNorm Bind Group Layout"),
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
            label: Some("AdaptiveInstanceNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: content_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: style_mean_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: style_std_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("AdaptiveInstanceNorm Shader"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("AdaptiveInstanceNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("AdaptiveInstanceNorm Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("AdaptiveInstanceNorm Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AdaptiveInstanceNorm Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            use crate::device::{DeviceCapabilities, WorkloadType};
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![batch, channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply adaptive instance normalization (`AdaIN`) for style transfer
    /// # Arguments
    /// * `style_mean` - Style mean tensor [C]
    /// * `style_std` - Style std tensor [C]
    /// # Errors
    /// Returns [`Err`] if content is not 4D, style mean/std shapes are invalid, channel count
    /// mismatches, buffer allocation fails, GPU dispatch fails, buffer readback fails, or the device is lost.
    pub fn adaptive_instance_norm(self, style_mean: Self, style_std: Self) -> Result<Self> {
        AdaptiveInstanceNorm::new(self, style_mean, style_std)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // No longer needed - using Tensor method API
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_adaptive_instance_norm_basic() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let content_data = vec![1.0; 3 * 4 * 4];
        let content = Tensor::new(content_data.clone(), vec![1, 3, 4, 4], dev.clone());
        let style_mean = Tensor::new(vec![0.5, 0.5, 0.5], vec![3], dev.clone());
        let style_std = Tensor::new(vec![0.2, 0.2, 0.2], vec![3], dev);
        let output_tensor = content
            .adaptive_instance_norm(style_mean, style_std)
            .unwrap();
        let output = output_tensor.to_vec().unwrap();
        assert_eq!(output.len(), content_data.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_adaptive_instance_norm_edge_cases() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with zero style std (should clamp)
        let content = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], dev.clone());
        let style_mean = Tensor::new(vec![0.0], vec![1], dev.clone());
        let style_std = Tensor::new(vec![0.0], vec![1], dev.clone());
        let output_tensor = content
            .adaptive_instance_norm(style_mean, style_std)
            .unwrap();
        let output = output_tensor.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));

        // Test with single channel, single pixel
        let content = Tensor::new(vec![5.0], vec![1, 1, 1, 1], dev.clone());
        let style_mean = Tensor::new(vec![1.0], vec![1], dev.clone());
        let style_std = Tensor::new(vec![2.0], vec![1], dev);
        let output_tensor = content
            .adaptive_instance_norm(style_mean, style_std)
            .unwrap();
        let output = output_tensor.to_vec().unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[tokio::test]
    async fn test_adaptive_instance_norm_boundary() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with different style statistics
        let content_data = vec![0.0, 1.0, 2.0, 3.0];

        // Style 1: mean=0, std=1
        let content1 = Tensor::new(content_data.clone(), vec![1, 1, 2, 2], dev.clone());
        let style_mean1 = Tensor::new(vec![0.0], vec![1], dev.clone());
        let style_std1 = Tensor::new(vec![1.0], vec![1], dev.clone());
        let result1 = content1
            .adaptive_instance_norm(style_mean1, style_std1)
            .unwrap();
        let output1 = result1.to_vec().unwrap();

        // Style 2: mean=10, std=5
        let content2 = Tensor::new(content_data, vec![1, 1, 2, 2], dev.clone());
        let style_mean2 = Tensor::new(vec![10.0], vec![1], dev.clone());
        let style_std2 = Tensor::new(vec![5.0], vec![1], dev);
        let result2 = content2
            .adaptive_instance_norm(style_mean2, style_std2)
            .unwrap();
        let output2 = result2.to_vec().unwrap();

        assert!(output1.iter().all(|&x| x.is_finite()));
        assert!(output2.iter().all(|&x| x.is_finite()));
        // Different style should produce different output
        assert_ne!(output1, output2);
        // Output2 should have higher values (mean=10)
        assert!(output2.iter().sum::<f32>() > output1.iter().sum::<f32>());
    }

    #[tokio::test]
    async fn test_adaptive_instance_norm_large_batch() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Multiple batches and channels
        let batch_size = 2;
        let channels = 4;
        let height = 8;
        let width = 8;

        let content_data: Vec<f32> = (0..batch_size * channels * height * width)
            .map(|i| (i % 10) as f32)
            .collect();
        let content = Tensor::new(
            content_data.clone(),
            vec![batch_size, channels, height, width],
            dev.clone(),
        );
        let style_mean = Tensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![channels], dev.clone());
        let style_std = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![channels], dev);

        let output_tensor = content
            .adaptive_instance_norm(style_mean, style_std)
            .unwrap();
        let output = output_tensor.to_vec().unwrap();

        assert_eq!(output.len(), content_data.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_adaptive_instance_norm_precision() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with known values for style transfer
        let content_data = vec![
            0.0, 1.0, 2.0, 3.0, // Mean = 1.5
        ];
        let content = Tensor::new(content_data, vec![1, 1, 2, 2], dev.clone());
        let style_mean = Tensor::new(vec![5.0], vec![1], dev.clone()); // Target mean
        let style_std = Tensor::new(vec![2.0], vec![1], dev); // Target std

        let result = content
            .adaptive_instance_norm(style_mean, style_std)
            .unwrap();
        let output = result.to_vec().unwrap();

        // After AdaIN, output should have approximately the target mean
        let out_mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!((out_mean - 5.0).abs() < 0.1);

        // Output should preserve relative relationships (normalized)
        assert!(output[0] < output[1]);
        assert!(output[1] < output[2]);
        assert!(output[2] < output[3]);
    }
}
