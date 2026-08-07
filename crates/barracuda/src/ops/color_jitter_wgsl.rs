// SPDX-License-Identifier: AGPL-3.0-or-later
//! Color Jitter - Image data augmentation - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Randomly adjusts image colors for data augmentation:
//! ```text
//! Input:  [B, C, H, W] image tensor
//! Output: [B, C, H, W] with adjusted:
//!   - Brightness (additive)
//!   - Contrast (multiplicative)
//!   - Saturation (color intensity)
//!   - Hue (color shift)
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Image color augmentation: brightness, contrast, saturation, hue.
pub struct ColorJitter {
    input: Tensor,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
}

impl ColorJitter {
    /// Create a color jitter operation. Expects [B, C, H, W] input.
    #[must_use]
    pub fn new(input: Tensor, brightness: f32, contrast: f32, saturation: f32, hue: f32) -> Self {
        Self {
            input,
            brightness,
            contrast,
            saturation,
            hue,
        }
    }

    /// Execute color jitter on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Expect 4D tensor [batch, channels, height, width]
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0, 0, 0],
                actual: shape.to_vec(),
            });
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let total_size = self.input.len();

        let output_buffer = device.create_buffer_f32(total_size)?;

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            height: u32,
            width: u32,
            brightness: f32,
            contrast: f32,
            saturation: f32,
            hue: f32,
        }

        let params_data = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            brightness: self.brightness,
            contrast: self.contrast,
            saturation: self.saturation,
            hue: self.hue,
        };
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        ComputeDispatch::new(device, "ColorJitter")
            .shader(
                include_str!("../shaders/augmentation/color_jitter_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply color jitter augmentation. Expects [B, C, H, W].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn color_jitter_wgsl(
        self,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
    ) -> Result<Self> {
        ColorJitter::new(self, brightness, contrast, saturation, hue).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_color_jitter_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Small 1x3x2x2 image (1 batch, 3 channels RGB, 2x2 pixels)
        let input_data = vec![
            // R channel
            0.5, 0.5, 0.5, 0.5, // G channel
            0.5, 0.5, 0.5, 0.5, // B channel
            0.5, 0.5, 0.5, 0.5,
        ];
        let input = Tensor::from_vec_on(input_data, vec![1, 3, 2, 2], device)
            .await
            .unwrap();

        let result = input.color_jitter_wgsl(0.1, 0.1, 0.1, 0.1).unwrap();
        let output = result.to_vec().unwrap();

        // Verify output is modified (not exact same values)
        // and values are in valid range [0, 1]
        assert_eq!(output.len(), 12);
        for &val in &output {
            assert!((0.0..=1.0).contains(&val));
        }
        // At least one value should be different due to jitter
        let all_same = output.iter().all(|&x| (x - 0.5).abs() < 0.01);
        assert!(!all_same, "Color jitter should modify values");
    }
}
