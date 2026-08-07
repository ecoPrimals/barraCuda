// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SeparableConv2D` - Depthwise Separable Convolution 2D
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Efficient convolution: depthwise followed by pointwise (1x1)
//! Used in `MobileNet`, Xception, `EfficientNet`
//!
//! Reduces parameters from `C_in`*`C_out`*K*K to `C_in`*K*K + `C_in`*`C_out`

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SeparableConv2DParams {
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
    mode: u32, // 0 = depthwise, 1 = pointwise
}

/// Depthwise separable 2D convolution (depthwise + pointwise).
pub struct SeparableConv2D {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    mode: SeparableConvMode,
}

/// Mode for separable convolution: depthwise (per-channel) or pointwise (1×1).
#[derive(Clone, Copy)]
pub enum SeparableConvMode {
    /// Depthwise: each channel convolved independently with its own kernel.
    Depthwise,
    /// Pointwise: 1×1 convolution mixing channels.
    Pointwise,
}

impl SeparableConv2D {
    /// Create a new separable convolution operation.
    /// # Errors
    /// Returns [`Err`] if input is not 4D, or `kernel_size` or stride is zero.
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        mode: SeparableConvMode,
    ) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "separable_conv2d",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if kernel_size == 0 || stride == 0 {
            return Err(BarracudaError::invalid_op(
                "separable_conv2d",
                "kernel_size and stride must be positive",
            ));
        }

        Ok(Self {
            input,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            mode,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/conv/separable_conv2d_f64.wgsl");
            SHADER
        }
    }

    /// Execute the separable convolution and return the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_height = ((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1;
        let out_width = ((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1;

        let (out_channels, output_size) = match self.mode {
            SeparableConvMode::Depthwise => (
                in_channels,
                batch_size * in_channels * out_height * out_width,
            ),
            SeparableConvMode::Pointwise => {
                let out_ch = self.weight.shape()[0];
                (out_ch, batch_size * out_ch * out_height * out_width)
            }
        };

        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = SeparableConv2DParams {
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
            mode: match self.mode {
                SeparableConvMode::Depthwise => 0,
                SeparableConvMode::Pointwise => 1,
            },
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("separable_conv2d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "separable_conv2d")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.weight.buffer())
            .storage_read(2, self.bias.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(
                (out_width as u32).div_ceil(16),
                (out_height as u32).div_ceil(16),
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
    /// Apply separable convolution 2D
    /// # Arguments
    /// - `weight`: Weight tensor
    /// - `bias`: Bias tensor
    /// - `kernel_size`: Kernel size
    /// - `stride`: Stride
    /// - `padding`: Padding
    /// - `mode`: Depthwise or Pointwise
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn separable_conv2d(
        self,
        weight: Self,
        bias: Self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        mode: SeparableConvMode,
    ) -> Result<Self> {
        SeparableConv2D::new(self, weight, bias, kernel_size, stride, padding, mode)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_separable_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        // Depthwise weight: [C, 1, K, K] = [3, 1, 3, 3] = 27 elements
        let weight = Tensor::from_vec_on(vec![0.1; 3 * 3 * 3], vec![3, 1, 3, 3], device.clone())
            .await
            .unwrap();
        let bias = Tensor::from_vec_on(vec![0.0; 3], vec![3], device.clone())
            .await
            .unwrap();

        let output = input
            .separable_conv2d(weight, bias, 3, 1, 1, SeparableConvMode::Depthwise)
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 3);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
