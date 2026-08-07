// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DilatedConv2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    out_height: u32,
    out_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    _padding: u32,
}

/// 2D convolution with dilation (atrous convolution).
pub struct DilatedConv2D {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl DilatedConv2D {
    /// Creates a new dilated conv2d. Input must be 4D [B, C, H, W].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Self> {
        // Validate shapes
        if input.shape().len() != 4 {
            return Err(BarracudaError::invalid_op(
                "dilated_conv2d",
                format!(
                    "Input must be 4D [B, C, H, W], got shape: {:?}",
                    input.shape()
                ),
            ));
        }

        if weight.shape().len() != 4 {
            return Err(BarracudaError::invalid_op(
                "dilated_conv2d",
                format!(
                    "Weight must be 4D [C_out, C_in, Kh, Kw], got shape: {:?}",
                    weight.shape()
                ),
            ));
        }

        Ok(Self {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    /// Executes dilated conv2d and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let in_shape = self.input.shape();
        let w_shape = self.weight.shape();

        let batch_size = in_shape[0];
        let in_channels = in_shape[1];
        let in_height = in_shape[2];
        let in_width = in_shape[3];

        let out_channels = w_shape[0];
        let kernel_height = w_shape[2];
        let kernel_width = w_shape[3];

        // Calculate output dimensions
        let out_height =
            (in_height + 2 * self.padding.0 - self.dilation.0 * (kernel_height - 1) - 1)
                / self.stride.0
                + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.dilation.1 * (kernel_width - 1) - 1)
            / self.stride.1
            + 1;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create bias buffer (or zeros)
        let zeros_buffer;
        let bias_buffer = if let Some(ref bias) = self.bias {
            bias.buffer()
        } else {
            let zeros = vec![0.0f32; out_channels];
            zeros_buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dilated Conv2D Bias (zeros)"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            &zeros_buffer
        };

        let params = DilatedConv2DParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            kernel_height: kernel_height as u32,
            kernel_width: kernel_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            stride_h: self.stride.0 as u32,
            stride_w: self.stride.1 as u32,
            pad_h: self.padding.0 as u32,
            pad_w: self.padding.1 as u32,
            dilation_h: self.dilation.0 as u32,
            dilation_w: self.dilation.1 as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dilated Conv2D Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "DilatedConv2D")
            .shader(
                include_str!("../shaders/conv/dilated_conv2d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.weight.buffer())
            .storage_read(2, bias_buffer)
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(
                (out_width as u32).div_ceil(16),
                (out_height as u32).div_ceil(16),
                batch_size as u32 * out_channels as u32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dilated_conv2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 8;
        let height = 32;
        let width = 32;
        let kernel_size = 3;

        let input_data = vec![1.0; batch_size * in_channels * height * width];
        let weight_data = vec![0.1; out_channels * in_channels * kernel_size * kernel_size];
        let bias_data = vec![0.0; out_channels];

        let input = Tensor::from_vec_on(
            input_data,
            vec![batch_size, in_channels, height, width],
            device.clone(),
        )
        .await
        .unwrap();

        let weight = Tensor::from_vec_on(
            weight_data,
            vec![out_channels, in_channels, kernel_size, kernel_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias = Tensor::from_vec_on(bias_data, vec![out_channels], device.clone())
            .await
            .unwrap();

        let output = DilatedConv2D::new(
            input,
            weight,
            Some(bias),
            (1, 1), // stride
            (1, 1), // padding
            (2, 2), // dilation = 2
        )
        .unwrap()
        .execute()
        .unwrap();

        // With dilation=2, kernel 3x3: effective kernel = 5x5
        // out_h = (32 + 2*pad - dilation*(kernel_size-1) - 1)/stride + 1 = 30
        let pad = 1;
        let stride = 1;
        let expected_h = (height + 2 * pad - 2 * (kernel_size - 1) - 1) / stride + 1;
        let expected_w = (width + 2 * pad - 2 * (kernel_size - 1) - 1) / stride + 1;
        assert_eq!(
            output.shape(),
            &[batch_size, out_channels, expected_h, expected_w]
        );
    }
}
