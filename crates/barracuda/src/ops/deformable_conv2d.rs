// SPDX-License-Identifier: AGPL-3.0-or-later
//! `DeformableConv2D` - Deformable Convolution 2D
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Convolution with learnable offsets for sampling positions
//! Adapts receptive field based on content
//!
//! Reference: "Deformable Convolutional Networks" by Dai et al. (2017)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DeformableConv2DParams {
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
    dilation: u32,
    deform_groups: u32,
}

/// Deformable 2D convolution with learnable sampling offsets.
pub struct DeformableConv2D {
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    deform_groups: usize,
}

impl DeformableConv2D {
    /// Creates a new deformable conv2d. Input must be 4D [B, C, H, W].
    /// # Errors
    /// Returns [`Err`] if input is not 4D [B, C, H, W] or `kernel_size`, stride, or dilation is zero.
    pub fn new(
        input: Tensor,
        offset: Tensor,
        weight: Tensor,
        bias: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        deform_groups: usize,
    ) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "deformable_conv2d",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if kernel_size == 0 || stride == 0 || dilation == 0 {
            return Err(BarracudaError::invalid_op(
                "deformable_conv2d",
                "kernel_size, stride, and dilation must be positive",
            ));
        }

        Ok(Self {
            input,
            offset,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            deform_groups,
        })
    }

    /// Executes deformable conv2d and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_height =
            ((in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride)
                + 1;
        let out_width =
            ((in_width + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride)
                + 1;
        let out_channels = self.bias.shape()[0];

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = DeformableConv2DParams {
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
            dilation: self.dilation as u32,
            deform_groups: self.deform_groups as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("deformable_conv2d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "DeformableConv2D")
            .shader(
                include_str!("../shaders/conv/deformable_conv2d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.offset.buffer())
            .storage_read(2, self.weight.buffer())
            .storage_read(3, self.bias.buffer())
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(
                (out_width as u32).div_ceil(16),
                (out_height as u32).div_ceil(16),
                (batch_size * out_channels) as u32,
            )
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply deformable convolution 2D
    /// # Arguments
    /// - `offset`: Learned offset tensor [B, 2*K*K, `H_out`, `W_out`]
    /// - `weight`: Weight tensor [`C_out`, `C_in`, K, K]
    /// - `bias`: Bias tensor [`C_out`]
    /// - `kernel_size`: Kernel size
    /// - `stride`: Stride
    /// - `padding`: Padding
    /// - `dilation`: Dilation
    /// - `deform_groups`: Number of deformable groups
    /// # Errors
    /// Returns [`Err`] if input is not 4D, `kernel_size/stride/dilation` is zero, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn deformable_conv2d(
        self,
        offset: Self,
        weight: Self,
        bias: Self,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        deform_groups: usize,
    ) -> Result<Self> {
        DeformableConv2D::new(
            self,
            offset,
            weight,
            bias,
            kernel_size,
            stride,
            padding,
            dilation,
            deform_groups,
        )?
        .execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_deformable_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        let offset = Tensor::from_vec_on(vec![0.0; 18 * 2 * 2], vec![1, 18, 2, 2], device.clone())
            .await
            .unwrap();
        let weight =
            Tensor::from_vec_on(vec![0.1; 4 * 3 * 3 * 3], vec![4, 3, 3, 3], device.clone())
                .await
                .unwrap();
        let bias = Tensor::from_vec_on(vec![0.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let output = input
            .deformable_conv2d(offset, weight, bias, 3, 1, 1, 1, 1)
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape()[0], 1);
        assert_eq!(output.shape()[1], 4);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
