// SPDX-License-Identifier: AGPL-3.0-or-later
//! SSIM - Structural Similarity Index (Wang et al.)
//!
//! Perceptual similarity metric for images.
//! Considers luminance, contrast, and structure.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// SSIM operation
pub struct SSIM {
    image1: Tensor,
    image2: Tensor,
    window_size: usize,
    c1: f32,
    c2: f32,
}

impl SSIM {
    /// Create a new SSIM operation
    /// # Errors
    /// Returns [`Err`] if image shapes differ, images are not 2D, or `window_size` is invalid.
    pub fn new(
        image1: Tensor,
        image2: Tensor,
        window_size: usize,
        c1: f32,
        c2: f32,
    ) -> Result<Self> {
        let shape1 = image1.shape();
        let shape2 = image2.shape();

        if shape1 != shape2 {
            return Err(crate::error::BarracudaError::invalid_op(
                "SSIM",
                format!("Images must have same shape: {shape1:?} vs {shape2:?}"),
            ));
        }

        if shape1.len() != 2 {
            return Err(crate::error::BarracudaError::invalid_op(
                "SSIM",
                format!("Expected 2D tensor (H, W), got {}D", shape1.len()),
            ));
        }

        if window_size == 0 || window_size > shape1[0] || window_size > shape1[1] {
            return Err(crate::error::BarracudaError::invalid_op(
                "SSIM",
                format!("Invalid window_size: {window_size}"),
            ));
        }

        Ok(Self {
            image1,
            image2,
            window_size,
            c1,
            c2,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!(
                    "../shaders/misc/ssim_f64.wgsl"
                ).to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the SSIM operation
    /// # Errors
    /// Returns [`Err`] if window size is too large for image, or buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<f32> {
        let device = self.image1.device();
        let shape = self.image1.shape();

        let height = shape[0];
        let width = shape[1];
        let num_windows_x = width - self.window_size + 1;
        let num_windows_y = height - self.window_size + 1;
        let num_windows = num_windows_x * num_windows_y;

        if num_windows == 0 {
            return Err(crate::error::BarracudaError::invalid_op(
                "SSIM",
                "Window size too large for image dimensions",
            ));
        }

        // Create buffers
        let image1_buffer = self.image1.buffer();
        let image2_buffer = self.image2.buffer();

        let window_ssim_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSIM Window Output"),
            size: (num_windows * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
            window_size: u32,
            c1: f32,
            c2: f32,
            num_windows: u32,
        }

        let params = Params {
            width: width as u32,
            height: height as u32,
            window_size: self.window_size as u32,
            c1: self.c1,
            c2: self.c2,
            num_windows: num_windows as u32,
        };

        let params_buffer = device.create_uniform_buffer("SSIM Params", &params);

        let workgroups_x = (num_windows_x as u32).div_ceil(16);
        let workgroups_y = (num_windows_y as u32).div_ceil(16);

        ComputeDispatch::new(device, "SSIM")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, image1_buffer)
            .storage_read(1, image2_buffer)
            .storage_rw(2, &window_ssim_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        // Read back results and compute mean SSIM
        let window_ssim_data = crate::utils::read_buffer(device, &window_ssim_buffer, num_windows)?;
        let ssim_sum: f32 = window_ssim_data.iter().sum();
        Ok(ssim_sum / num_windows as f32)
    }
}

impl Tensor {
    /// Compute SSIM between two images
    /// # Arguments
    /// * `other` - Second image tensor (must have same shape)
    /// * `window_size` - Size of sliding window (typically 11)
    /// * `c1` - Stability constant for luminance (typically 0.01^2)
    /// * `c2` - Stability constant for contrast (typically 0.03^2)
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn ssim(self, other: Tensor, window_size: usize, c1: f32, c2: f32) -> Result<f32> {
        SSIM::new(self, other, window_size, c1, c2)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_ssim_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let image1 = Tensor::new(vec![0.5; 64 * 64], vec![64, 64], device.clone());
        let image2 = Tensor::new(vec![0.5; 64 * 64], vec![64, 64], device.clone());
        let similarity = image1.ssim(image2, 11, 0.01, 0.03).unwrap();
        assert!(similarity.is_finite());
        assert!(similarity > 0.9); // Should be close to 1.0 for identical images
    }

    #[tokio::test]
    async fn test_ssim_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Small image
        let image1 = Tensor::new(vec![0.5; 16 * 16], vec![16, 16], device.clone());
        let image2 = Tensor::new(vec![0.5; 16 * 16], vec![16, 16], device.clone());
        let similarity = image1.ssim(image2, 5, 0.01, 0.03).unwrap();
        assert!(similarity.is_finite());

        // Different images
        let image1 = Tensor::new(vec![0.0; 32 * 32], vec![32, 32], device.clone());
        let image2 = Tensor::new(vec![1.0; 32 * 32], vec![32, 32], device.clone());
        let similarity = image1.ssim(image2, 7, 0.01, 0.03).unwrap();
        assert!(similarity < 1.0);
    }
}
