// SPDX-License-Identifier: AGPL-3.0-or-later
//! `OctaveConv2D` - Octave Convolution 2D
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

use crate::device::capabilities::WORKGROUP_SIZE_2D;
use crate::device::compute_pipeline::ComputeDispatch;
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

/// Octave convolution 2D operator (high/low frequency paths).
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

/// Convolution path for octave convolution (high/low frequency routing).
#[derive(Clone, Copy)]
pub enum OctaveConvPath {
    /// High-frequency input → high-frequency output.
    HighToHigh,
    /// High-frequency input → low-frequency output (downsample).
    HighToLow,
    /// Low-frequency input → high-frequency output (upsample).
    LowToHigh,
    /// Low-frequency input → low-frequency output.
    LowToLow,
}

impl OctaveConv2D {
    /// Create octave conv2d with inputs, weights, and path.
    /// # Errors
    /// Returns [`Err`] if `kernel_size` or stride is zero, or if `input_high/input_low` is missing for the given path.
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
        include_str!("../shaders/conv/octave_conv2d_f64.wgsl")
    }

    /// Run octave convolution on GPU.
    /// # Errors
    /// Returns [`Err`] if no input provided, buffer allocation fails, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = match (&self.input_high, &self.input_low) {
            (Some(h), _) => h.device(),
            (_, Some(l)) => l.device(),
            _ => {
                return Err(BarracudaError::invalid_op(
                    "octave_conv2d",
                    "No input provided",
                ));
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

        let placeholder = device.placeholder_buffer();
        let input_high_buffer = self
            .input_high
            .as_ref()
            .map_or(placeholder, super::super::tensor::Tensor::buffer);
        let input_low_buffer = self
            .input_low
            .as_ref()
            .map_or(placeholder, super::super::tensor::Tensor::buffer);

        ComputeDispatch::new(device, "octave_conv2d")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_high_buffer)
            .storage_read(1, input_low_buffer)
            .storage_read(2, self.weight.buffer())
            .storage_read(3, self.bias.buffer())
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(
                (out_width as u32).div_ceil(WORKGROUP_SIZE_2D),
                (out_height as u32).div_ceil(WORKGROUP_SIZE_2D),
                (batch_size * out_channels) as u32,
            )
            .submit()?;

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
    /// # Arguments
    /// - `input_high`: High frequency input [B, `C_H`, H, W] (optional)
    /// - `input_low`: Low frequency input [B, `C_L`, H/2, W/2] (optional)
    /// - `weight`: Weight tensor
    /// - `bias`: Bias tensor
    /// - `kernel_size`: Kernel size
    /// - `stride`: Stride
    /// - `padding`: Padding
    /// - `path`: Convolution path (H→H, H→L, L→H, L→L)
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn octave_conv2d(
        self,
        input_high: Option<Self>,
        input_low: Option<Self>,
        weight: Self,
        bias: Self,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_octave_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
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
