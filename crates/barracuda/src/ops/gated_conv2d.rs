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

use crate::device::compute_pipeline::ComputeDispatch;
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

        ComputeDispatch::new(device, "GatedConv2D")
            .shader(
                include_str!("../shaders/conv/gated_conv2d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.weight_feature.buffer())
            .storage_read(2, self.weight_gate.buffer())
            .storage_read(3, self.bias_feature.buffer())
            .storage_read(4, self.bias_gate.buffer())
            .storage_rw(5, &output_buffer)
            .uniform(6, &params_buffer)
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
