// SPDX-License-Identifier: AGPL-3.0-or-later
//! PSNR - Peak Signal-to-Noise Ratio
//!
//! Measures reconstruction quality in dB.
//! Higher is better (typically 30-50 dB for good quality).
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

/// PSNR operation
pub struct PSNR {
    original: Tensor,
    reconstructed: Tensor,
    max_pixel_value: f32,
}

impl PSNR {
    /// Create a new PSNR operation
    /// # Errors
    /// Returns [`Err`] if tensor shapes do not match or tensors are empty.
    pub fn new(original: Tensor, reconstructed: Tensor, max_pixel_value: f32) -> Result<Self> {
        let shape1 = original.shape();
        let shape2 = reconstructed.shape();

        if shape1 != shape2 {
            return Err(crate::error::BarracudaError::invalid_op(
                "PSNR",
                format!("Tensors must have same shape: {shape1:?} vs {shape2:?}"),
            ));
        }

        if original.is_empty() {
            return Err(crate::error::BarracudaError::invalid_op(
                "PSNR",
                "Empty tensors",
            ));
        }

        Ok(Self {
            original,
            reconstructed,
            max_pixel_value,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/misc/psnr_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the PSNR operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<f32> {
        let device = self.original.device();
        let size = self.original.len();

        let mse_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            max_pixel_value: f32,
        }

        let params = Params {
            size: size as u32,
            max_pixel_value: self.max_pixel_value,
        };

        let params_buffer = device.create_uniform_buffer("PSNR Params", &params);

        ComputeDispatch::new(device, "PSNR")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.original.buffer())
            .storage_read(1, self.reconstructed.buffer())
            .storage_rw(2, &mse_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results and compute MSE
        let mse_data = crate::utils::read_buffer(device, &mse_buffer, size)?;
        let mse = mse_data.iter().sum::<f32>() / size as f32;

        if mse < 1e-10 {
            return Ok(f32::INFINITY); // Perfect reconstruction
        }

        // PSNR = 10 * log10(MAX^2 / MSE)
        let psnr_val = 10.0 * (self.max_pixel_value * self.max_pixel_value / mse).log10();

        Ok(psnr_val)
    }
}

impl Tensor {
    /// Compute PSNR between two tensors
    /// # Arguments
    /// * `other` - Reconstructed tensor (must have same shape)
    /// * `max_pixel_value` - Maximum pixel value (typically 1.0 or 255.0)
    /// # Errors
    /// Returns [`Err`] if shapes do not match, tensors are empty, or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn psnr(self, other: Tensor, max_pixel_value: f32) -> Result<f32> {
        PSNR::new(self, other, max_pixel_value)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_psnr_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let original = Tensor::new(vec![0.5; 1000], vec![1000], device.clone());
        let reconstructed = Tensor::new(vec![0.5; 1000], vec![1000], device);
        let psnr_val = original.psnr(reconstructed, 1.0).unwrap();
        assert!(psnr_val > 100.0); // Should be very high for identical signals
    }

    #[tokio::test]
    async fn test_psnr_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Perfect reconstruction
        let original = Tensor::new(vec![0.1, 0.5, 0.9], vec![3], device.clone());
        let reconstructed = Tensor::new(vec![0.1, 0.5, 0.9], vec![3], device.clone());
        let psnr_val = original.psnr(reconstructed, 1.0).unwrap();
        assert!(psnr_val.is_infinite()); // MSE ~= 0

        // Significant difference (low PSNR)
        let original = Tensor::new(vec![1.0; 100], vec![100], device.clone());
        let reconstructed = Tensor::new(vec![0.5; 100], vec![100], device);
        let psnr_val = original.psnr(reconstructed, 1.0).unwrap();
        assert!(psnr_val.is_finite());
        assert!(psnr_val < 10.0); // Poor quality
    }
}
