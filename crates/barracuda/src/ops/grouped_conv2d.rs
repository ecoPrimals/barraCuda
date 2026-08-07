// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grouped `Conv2D` - Convolution with channel groups (Pure WGSL)
//!
//! Divides input/output channels into groups, reducing parameters
//! Used in `ResNeXt`, `ShuffleNet`, `MobileNet` architectures
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Grouped 2D Convolution
pub struct GroupedConv2D {
    input: Tensor,
    kernel: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl GroupedConv2D {
    /// Create grouped 2D convolution (channels split into independent groups).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        kernel: Tensor,
        stride: usize,
        padding: usize,
        groups: usize,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "grouped_conv2d",
                "input must be 4D [batch, channels, height, width]",
            ));
        }

        let kernel_shape = kernel.shape();
        if kernel_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "grouped_conv2d",
                "kernel must be 4D [out_channels, in_channels_per_group, kernel_h, kernel_w]",
            ));
        }

        let _ = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let out_channels = kernel_shape[0];
        let in_per_group = kernel_shape[1];
        let kernel_size = kernel_shape[2]; // Assuming square kernel

        if !in_channels.is_multiple_of(groups) {
            return Err(BarracudaError::invalid_op(
                "grouped_conv2d",
                "in_channels must be divisible by groups",
            ));
        }

        if !out_channels.is_multiple_of(groups) {
            return Err(BarracudaError::invalid_op(
                "grouped_conv2d",
                "out_channels must be divisible by groups",
            ));
        }

        if in_channels / groups != in_per_group {
            return Err(BarracudaError::invalid_op(
                "grouped_conv2d",
                "kernel in_channels_per_group must match in_channels / groups",
            ));
        }

        if let Some(ref bias_tensor) = bias {
            let bias_size = bias_tensor.shape().iter().product::<usize>();
            if bias_size != out_channels {
                return Err(BarracudaError::invalid_op(
                    "grouped_conv2d",
                    "bias must have out_channels elements",
                ));
            }
        }

        let _ = (
            (in_height + 2 * padding - kernel_size) / stride + 1,
            (in_width + 2 * padding - kernel_size) / stride + 1,
        );

        Ok(Self {
            input,
            kernel,
            bias,
            stride,
            padding,
            groups,
        })
    }

    /// Execute grouped convolution.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        let kernel_shape = self.kernel.shape();
        let out_channels = kernel_shape[0];
        let in_per_group = kernel_shape[1];
        let kernel_size = kernel_shape[2];

        let out_height = (in_height + 2 * self.padding - kernel_size) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - kernel_size) / self.stride + 1;
        let out_per_group = out_channels / self.groups;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create bias buffer - use tensor buffer directly or create zero buffer
        let zero_bias;
        let bias_buffer: &wgpu::Buffer = if let Some(ref bias_tensor) = self.bias {
            bias_tensor.buffer()
        } else {
            // Create zero-initialized buffer for bias
            let zeros = vec![0.0f32; out_channels];
            zero_bias = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GroupedConv2D Bias Zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            &zero_bias
        };

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
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
            groups: u32,
            in_per_group: u32,
            out_per_group: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_size: kernel_size as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            groups: self.groups as u32,
            in_per_group: in_per_group as u32,
            out_per_group: out_per_group as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GroupedConv2D Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "GroupedConv2D")
            .shader(
                include_str!("../shaders/conv/grouped_conv2d_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_read(2, self.kernel.buffer())
            .storage_read(3, bias_buffer)
            .storage_rw(4, &output_buffer)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_grouped_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 1;
        let in_channels = 8;
        let in_height = 32;
        let in_width = 32;
        let out_channels = 16;
        let kernel_size = 3;
        let groups = 2;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * in_channels * in_height * in_width],
            vec![batch_size, in_channels, in_height, in_width],
            device.clone(),
        )
        .await
        .unwrap();

        let in_per_group = in_channels / groups;
        let kernel = Tensor::from_vec_on(
            vec![0.1; out_channels * in_per_group * kernel_size * kernel_size],
            vec![out_channels, in_per_group, kernel_size, kernel_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias = Tensor::from_vec_on(vec![0.0; out_channels], vec![out_channels], device.clone())
            .await
            .unwrap();

        let conv = GroupedConv2D::new(input, kernel, 1, 1, groups, Some(bias)).unwrap();
        let output = conv.execute().unwrap();

        assert_eq!(output.shape().len(), 4);
    }

    #[tokio::test]
    async fn test_grouped_conv2d_no_bias() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 1;
        let in_channels = 4;
        let in_height = 16;
        let in_width = 16;
        let out_channels = 8;
        let kernel_size = 3;
        let groups = 2;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * in_channels * in_height * in_width],
            vec![batch_size, in_channels, in_height, in_width],
            device.clone(),
        )
        .await
        .unwrap();

        let in_per_group = in_channels / groups;
        let kernel = Tensor::from_vec_on(
            vec![0.1; out_channels * in_per_group * kernel_size * kernel_size],
            vec![out_channels, in_per_group, kernel_size, kernel_size],
            device.clone(),
        )
        .await
        .unwrap();

        let conv = GroupedConv2D::new(input, kernel, 1, 1, groups, None).unwrap();
        let output = conv.execute().unwrap();

        assert_eq!(output.shape().len(), 4);
    }
}
