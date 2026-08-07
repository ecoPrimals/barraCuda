// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PixelShuffle` - Pixel Shuffle (Depth to Space)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Rearranges elements in a tensor from depth to spatial dimensions
//! Used in super-resolution networks (ESPCN, EDSR)
//!
//! Transform [B, C*r^2, H, W] → [B, C, H*r, W*r]

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PixelShuffleParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    upscale_factor: u32,
}

/// Pixel shuffle (depth to space) for super-resolution networks.
pub struct PixelShuffle {
    input: Tensor,
    upscale_factor: usize,
}

impl PixelShuffle {
    /// Creates a new pixel shuffle. Input channels must be divisible by `upscale_factor²`.
    /// # Errors
    /// Returns [`Err`] if input is not 4D, `upscale_factor` is zero, or input channels are not divisible by `upscale_factor²`.
    pub fn new(input: Tensor, upscale_factor: usize) -> Result<Self> {
        // Validate input shape: must be 4D [B, C*r^2, H, W]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "input must be 4D tensor [B, C*r^2, H, W]",
            ));
        }

        if upscale_factor == 0 {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "upscale_factor must be positive",
            ));
        }

        let in_channels = shape[1];
        if !in_channels.is_multiple_of(upscale_factor * upscale_factor) {
            return Err(BarracudaError::invalid_op(
                "pixel_shuffle",
                "input channels must be divisible by upscale_factor^2",
            ));
        }

        Ok(Self {
            input,
            upscale_factor,
        })
    }

    /// Executes pixel shuffle and returns the rearranged tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let in_channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let out_channels = in_channels / (self.upscale_factor * self.upscale_factor);
        let out_height = in_height * self.upscale_factor;
        let out_width = in_width * self.upscale_factor;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = PixelShuffleParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            upscale_factor: self.upscale_factor as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pixel_shuffle_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (out_width as u32).div_ceil(8);
        let workgroups_y = (out_height as u32).div_ceil(8);
        let workgroups_z = (batch_size * out_channels) as u32;

        ComputeDispatch::new(device, "PixelShuffle")
            .shader(include_str!("../shaders/misc/pixel_shuffle_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply pixel shuffle (depth to space)
    /// # Arguments
    /// - `upscale_factor`: Upscaling factor r (output will be H*r x W*r)
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn pixel_shuffle(self, upscale_factor: usize) -> Result<Self> {
        PixelShuffle::new(self, upscale_factor)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pixel_shuffle_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // [B=1, C*r^2=4, H=2, W=2] with r=2 → [B=1, C=1, H=4, W=4]
        let input_data = vec![1.0; 4 * 2 * 2];
        let input = Tensor::from_vec_on(input_data, vec![1, 4, 2, 2], device.clone())
            .await
            .unwrap();

        let output = input.pixel_shuffle(2).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
