// SPDX-License-Identifier: AGPL-3.0-or-later
//! OctaveConv2D - Octave Convolution 2D
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Multi-frequency convolution processing high and low frequency information separately
//! Reduces memory and computation while maintaining accuracy
//!
//! Reference: "Drop an Octave: Reducing Spatial Redundancy in CNNs with Octave Convolution" by Chen et al. (2019)

use crate::device::DeviceCapabilities;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OctaveConv2DParams {
    batch_size: u32,
    in_channels_high: u32,
    in_channels_low: u32,
    out_channels_high: u32,
    out_channels_low: u32,
    in_height_high: u32,
    in_width_high: u32,
    in_height_low: u32,
    in_width_low: u32,
    out_height_high: u32,
    out_width_high: u32,
    out_height_low: u32,
    out_width_low: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    path: u32, // 0=H→H, 1=H→L, 2=L→H, 3=L→L
}

pub struct OctaveConv2D {
    input_high: Option<Tensor>,
    input_low: Option<Tensor>,
    weight: Tensor,
    bias: Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    path: OctaveConvPath,
}

#[derive(Clone, Copy)]
pub enum OctaveConvPath {
    HighToHigh,
    HighToLow,
    LowToHigh,
    LowToLow,
}

impl OctaveConv2D {
    pub fn new(
        input_high: Option<Tensor>,
        input_low: Option<Tensor>,
        weight: Tensor,
        bias: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        path: OctaveConvPath,
    ) -> Result<Self> {
        if kernel_size == 0 || stride == 0 {
            return Err(BarracudaError::invalid_op(
                "octave_conv2d",
                "kernel_size and stride must be positive",
            ));
        }

        // Validate inputs based on path
        match path {
            OctaveConvPath::HighToHigh | OctaveConvPath::HighToLow => {
                if input_high.is_none() {
                    return Err(BarracudaError::invalid_op(
                        "octave_conv2d",
                        "input_high required for H→H and H→L paths",
                    ));
                }
            }
            OctaveConvPath::LowToHigh | OctaveConvPath::LowToLow => {
                if input_low.is_none() {
                    return Err(BarracudaError::invalid_op(
                        "octave_conv2d",
                        "input_low required for L→H and L→L paths",
                    ));
                }
            }
        }

        Ok(Self {
            input_high,
            input_low,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            path,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/conv/octave_conv2d_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = match (&self.input_high, &self.input_low) {
            (Some(h), _) => h.device(),
            (_, Some(l)) => l.device(),
            _ => {
                return Err(BarracudaError::invalid_op(
                    "octave_conv2d",
                    "No input provided",
                ))
            }
        };

        // Determine output dimensions based on path
        let (batch_size, out_channels, out_height, out_width) = match self.path {
            OctaveConvPath::HighToHigh | OctaveConvPath::LowToHigh => {
                if let Some(ref input_high) = self.input_high {
                    let shape = input_high.shape();
                    let h = shape[2];
                    let w = shape[3];
                    let out_h = ((h + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    let out_w = ((w + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    (shape[0], self.bias.shape()[0], out_h, out_w)
                } else if let Some(ref input_low) = self.input_low {
                    let shape = input_low.shape();
                    let h = shape[2];
                    let w = shape[3];
                    let out_h = ((h * 2 + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    let out_w = ((w * 2 + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    (shape[0], self.bias.shape()[0], out_h, out_w)
                } else {
                    return Err(BarracudaError::invalid_op(
                        "octave_conv2d",
                        "No input provided",
                    ));
                }
            }
            OctaveConvPath::HighToLow | OctaveConvPath::LowToLow => {
                if let Some(ref input_high) = self.input_high {
                    let shape = input_high.shape();
                    let h = shape[2];
                    let w = shape[3];
                    let out_h = ((h / 2 + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    let out_w = ((w / 2 + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    (shape[0], self.bias.shape()[0], out_h, out_w)
                } else if let Some(ref input_low) = self.input_low {
                    let shape = input_low.shape();
                    let h = shape[2];
                    let w = shape[3];
                    let out_h = ((h + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    let out_w = ((w + 2 * self.padding - self.kernel_size) / self.stride) + 1;
                    (shape[0], self.bias.shape()[0], out_h, out_w)
                } else {
                    return Err(BarracudaError::invalid_op(
                        "octave_conv2d",
                        "No input provided",
                    ));
                }
            }
        };

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Get input dimensions
        let (in_channels_high, in_height_high, in_width_high) =
            if let Some(ref input_high) = self.input_high {
                let shape = input_high.shape();
                (shape[1], shape[2], shape[3])
            } else {
                (0, 0, 0)
            };

        let (in_channels_low, in_height_low, in_width_low) =
            if let Some(ref input_low) = self.input_low {
                let shape = input_low.shape();
                (shape[1], shape[2], shape[3])
            } else {
                (0, 0, 0)
            };

        let params = OctaveConv2DParams {
            batch_size: batch_size as u32,
            in_channels_high: in_channels_high as u32,
            in_channels_low: in_channels_low as u32,
            out_channels_high: if matches!(
                self.path,
                OctaveConvPath::HighToHigh | OctaveConvPath::LowToHigh
            ) {
                out_channels as u32
            } else {
                0
            },
            out_channels_low: if matches!(
                self.path,
                OctaveConvPath::HighToLow | OctaveConvPath::LowToLow
            ) {
                out_channels as u32
            } else {
                0
            },
            in_height_high: in_height_high as u32,
            in_width_high: in_width_high as u32,
            in_height_low: in_height_low as u32,
            in_width_low: in_width_low as u32,
            out_height_high: if matches!(
                self.path,
                OctaveConvPath::HighToHigh | OctaveConvPath::LowToHigh
            ) {
                out_height as u32
            } else {
                0
            },
            out_width_high: if matches!(
                self.path,
                OctaveConvPath::HighToHigh | OctaveConvPath::LowToHigh
            ) {
                out_width as u32
            } else {
                0
            },
            out_height_low: if matches!(
                self.path,
                OctaveConvPath::HighToLow | OctaveConvPath::LowToLow
            ) {
                out_height as u32
            } else {
                0
            },
            out_width_low: if matches!(
                self.path,
                OctaveConvPath::HighToLow | OctaveConvPath::LowToLow
            ) {
                out_width as u32
            } else {
                0
            },
            kernel_size: self.kernel_size as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            path: match self.path {
                OctaveConvPath::HighToHigh => 0,
                OctaveConvPath::HighToLow => 1,
                OctaveConvPath::LowToHigh => 2,
                OctaveConvPath::LowToLow => 3,
            },
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("octave_conv2d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("octave_conv2d_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("octave_conv2d_bind_group_layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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
                    label: Some("octave_conv2d_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("octave_conv2d_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group entries - use dummy buffers for None inputs
        let dummy_buffer = device.create_buffer_f32(1)?;
        let input_high_buffer = self
            .input_high
            .as_ref()
            .map(|t| t.buffer())
            .unwrap_or(&dummy_buffer);
        let input_low_buffer = self
            .input_low
            .as_ref()
            .map(|t| t.buffer())
            .unwrap_or(&dummy_buffer);

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("octave_conv2d_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_high_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_low_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("octave_conv2d_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("octave_conv2d_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch using standard 2D shader workgroup size (16, 16)
            let caps = DeviceCapabilities::from_device(device);
            let (workgroups_x, workgroups_y) =
                caps.dispatch_2d(out_width as u32, out_height as u32);
            let workgroups_z = batch_size * out_channels;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z as u32);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply octave convolution 2D
    ///
    /// # Arguments
    /// - `input_high`: High frequency input [B, C_H, H, W] (optional)
    /// - `input_low`: Low frequency input [B, C_L, H/2, W/2] (optional)
    /// - `weight`: Weight tensor
    /// - `bias`: Bias tensor
    /// - `kernel_size`: Kernel size
    /// - `stride`: Stride
    /// - `padding`: Padding
    /// - `path`: Convolution path (H→H, H→L, L→H, L→L)
    pub fn octave_conv2d(
        self,
        input_high: Option<Tensor>,
        input_low: Option<Tensor>,
        weight: Tensor,
        bias: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        path: OctaveConvPath,
    ) -> Result<Self> {
        OctaveConv2D::new(
            input_high,
            input_low,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            path,
        )?
        .execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_octave_conv2d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_high =
            Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
                .await
                .unwrap();
        let weight =
            Tensor::from_vec_on(vec![0.1; 4 * 3 * 3 * 3], vec![4, 3, 3, 3], device.clone())
                .await
                .unwrap();
        let bias = Tensor::from_vec_on(vec![0.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let input_high_clone = input_high.clone();
        let output = input_high
            .octave_conv2d(
                Some(input_high_clone),
                None,
                weight,
                bias,
                3,
                1,
                1,
                OctaveConvPath::HighToHigh,
            )
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 4);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
