// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GatedConv2D` - Gated Convolution 2D
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Convolution with multiplicative gating mechanism
//! Used in `PixelCNN`, `WaveNet`, and generative models
//!
//! Output = `tanh(W_f` * x) ⊙ `sigmoid(W_g` * x)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GatedConv2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    _padding: u32,
    _padding2: u32,
}

/// Gated 2D convolution (PixelCNN/WaveNet style: `tanh(W_f`*x) ⊙ `sigmoid(W_g`*x)).
pub struct GatedConv2D {
    input: Tensor,
    weight_feature: Tensor,
    weight_gate: Tensor,
    bias_feature: Tensor,
    bias_gate: Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl GatedConv2D {
    /// Creates a new gated conv2d. Input must be 4D [B, C, H, W].
    /// # Errors
    /// Returns [`Err`] if input is not 4D [B, C, H, W], or if `kernel_size` or stride is zero.
    pub fn new(
        input: Tensor,
        weight_feature: Tensor,
        weight_gate: Tensor,
        bias_feature: Tensor,
        bias_gate: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "gated_conv2d",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if kernel_size == 0 || stride == 0 {
            return Err(BarracudaError::invalid_op(
                "gated_conv2d",
                "kernel_size and stride must be positive",
            ));
        }

        Ok(Self {
            input,
            weight_feature,
            weight_gate,
            bias_feature,
            bias_gate,
            kernel_size,
            stride,
            padding,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/conv/gated_conv2d_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Executes gated conv2d and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_height = ((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1;
        let out_width = ((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1;
        let out_channels = self.bias_feature.shape()[0];

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = GatedConv2DParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_size: self.kernel_size as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            _padding: 0,
            _padding2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gated_conv2d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("gated_conv2d_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("gated_conv2d_bind_group_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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
                    label: Some("gated_conv2d_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gated_conv2d_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gated_conv2d_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight_feature.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight_gate.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bias_feature.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.bias_gate.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("gated_conv2d_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gated_conv2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);
            let workgroups_z = batch_size * out_channels;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply gated convolution 2D
    /// # Arguments
    /// - `weight_feature`: Feature weight tensor [`C_out`, `C_in`, K, K]
    /// - `weight_gate`: Gate weight tensor [`C_out`, `C_in`, K, K]
    /// - `bias_feature`: Feature bias tensor [`C_out`]
    /// - `bias_gate`: Gate bias tensor [`C_out`]
    /// - `kernel_size`: Kernel size
    /// - `stride`: Stride
    /// - `padding`: Padding
    /// # Errors
    /// Returns [`Err`] if input is not 4D, `kernel_size/stride` is zero, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn gated_conv2d(
        self,
        weight_feature: Self,
        weight_gate: Self,
        bias_feature: Self,
        bias_gate: Self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        GatedConv2D::new(
            self,
            weight_feature,
            weight_gate,
            bias_feature,
            bias_gate,
            kernel_size,
            stride,
            padding,
        )?
        .execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gated_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        let weight_feature =
            Tensor::from_vec_on(vec![0.1; 4 * 3 * 3 * 3], vec![4, 3, 3, 3], device.clone())
                .await
                .unwrap();
        let weight_gate =
            Tensor::from_vec_on(vec![0.1; 4 * 3 * 3 * 3], vec![4, 3, 3, 3], device.clone())
                .await
                .unwrap();
        let bias_feature = Tensor::from_vec_on(vec![0.0; 4], vec![4], device.clone())
            .await
            .unwrap();
        let bias_gate = Tensor::from_vec_on(vec![0.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let output = input
            .gated_conv2d(
                weight_feature,
                weight_gate,
                bias_feature,
                bias_gate,
                3,
                1,
                1,
            )
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 4);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
