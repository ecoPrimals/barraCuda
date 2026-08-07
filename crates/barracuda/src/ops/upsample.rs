// SPDX-License-Identifier: AGPL-3.0-or-later
//! Upsample - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Upsample operation
pub struct Upsample {
    input: Tensor,
    size: Option<(usize, usize)>,
    scale_factor: Option<(f32, f32)>,
    mode: UpsampleMode,
    align_corners: bool,
}

/// Upsampling interpolation mode.
#[derive(Debug, Clone, Copy)]
pub enum UpsampleMode {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
}

impl Upsample {
    /// Create a new upsample operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if input is not 4D [B, C, H, W], or neither size nor `scale_factor` is provided.
    pub fn new(
        input: Tensor,
        size: Option<(usize, usize)>,
        scale_factor: Option<(f32, f32)>,
        mode: UpsampleMode,
        align_corners: bool,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Upsample expects 4D tensor [B, C, H, W], got shape {shape:?}"
            )));
        }

        if size.is_none() && scale_factor.is_none() {
            return Err(crate::error::BarracudaError::invalid_input(
                "Either size or scale_factor must be provided",
            ));
        }

        Ok(Self {
            input,
            size,
            scale_factor,
            mode,
            align_corners,
        })
    }

    /// Get the WGSL shader source
    /// Execute the upsample operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        // Compute output size
        let (out_height, out_width) = if let Some((h, w)) = self.size {
            (h, w)
        } else if let Some((sh, sw)) = self.scale_factor {
            (
                (in_height as f32 * sh) as usize,
                (in_width as f32 * sw) as usize,
            )
        } else {
            return Err(crate::error::BarracudaError::invalid_input(
                "Either size or scale_factor must be provided",
            ));
        };

        let output_size = batch_size * channels * out_height * out_width;

        // Access input buffer directly (zero-copy)
        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            in_height: u32,
            in_width: u32,
            out_height: u32,
            out_width: u32,
            mode: u32,
            align_corners: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            mode: match self.mode {
                UpsampleMode::Nearest => 0,
                UpsampleMode::Bilinear => 1,
            },
            align_corners: u32::from(self.align_corners),
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Upsample Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (out_width as u32).div_ceil(8);
        let workgroups_y = (out_height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "Upsample")
            .shader(include_str!("../shaders/misc/upsample_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsample_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 3, 4], device).unwrap();

        let upsampled = Upsample::new(input, Some((6, 8)), None, UpsampleMode::Nearest, false)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 6, 8]);
    }

    #[tokio::test]
    async fn test_upsample_scale_factor() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 2, 4], device).unwrap();

        let upsampled = Upsample::new(input, None, Some((2.0, 2.0)), UpsampleMode::Bilinear, false)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 4, 8]);
    }

    #[tokio::test]
    async fn test_upsample_invalid_shape() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Upsample::new(input, Some((10, 10)), None, UpsampleMode::Nearest, false,).is_err());
    }

    #[tokio::test]
    async fn test_upsample_no_params() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0; 12], vec![1, 1, 3, 4], device).unwrap();

        assert!(Upsample::new(input, None, None, UpsampleMode::Nearest, false,).is_err());
    }

    #[tokio::test]
    async fn test_upsample_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 16, 16], device).unwrap();

        let upsampled = Upsample::new(input, Some((32, 32)), None, UpsampleMode::Bilinear, true)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(upsampled.shape(), &vec![1, 1, 32, 32]);
    }
}
