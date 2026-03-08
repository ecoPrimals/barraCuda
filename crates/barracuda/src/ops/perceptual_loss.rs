// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PerceptualLoss` - Feature-based perceptual loss
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! Compares high-level features instead of pixels.
//! Used in style transfer and super-resolution.

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Perceptual Loss operation
pub struct PerceptualLoss {
    features1: Tensor,
    features2: Tensor,
    weights: Option<Tensor>,
}

impl PerceptualLoss {
    /// Create a new perceptual loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(features1: Tensor, features2: Tensor, weights: Option<Tensor>) -> Result<Self> {
        // Validate feature dimensions match
        if features1.shape() != features2.shape() {
            return Err(BarracudaError::shape_mismatch(
                features1.shape().to_vec(),
                features2.shape().to_vec(),
            ));
        }

        // Validate weights if provided
        if let Some(ref w) = weights {
            let features_size: usize = features1.shape().iter().product();
            let weights_size: usize = w.shape().iter().product();
            if !features_size.is_multiple_of(weights_size) {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "Weights dimension mismatch: features size {features_size} must be divisible by weights size {weights_size}"
                    ),
                });
            }
        }

        Ok(Self {
            features1,
            features2,
            weights,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/loss/perceptual_loss_f64.wgsl"
            ).to_string()
        });
        &SHADER
    }

    /// Execute the perceptual loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.features1.device();
        let size = self.features1.len();

        let num_workgroups = (size as u32).div_ceil(crate::device::capabilities::WORKGROUP_SIZE_1D);

        // Create reduction buffer (one slot per workgroup) and output buffer
        let loss_buffer = device.create_buffer_f32(num_workgroups as usize)?;
        let output_buffer = device.create_buffer_f32(1)?;
        device.write_buffer_f32(&loss_buffer, &vec![0.0; num_workgroups as usize])?;

        // Determine if weights are provided and number of weight groups
        let has_weights = self.weights.is_some() as u32;
        let num_weights = self
            .weights
            .as_ref()
            .map_or(0, |w| w.shape().iter().product::<usize>()) as u32;

        // Sentinel buffer when weights are None (keeps it alive for bind group creation)
        let sentinel_buf;
        let weights_buffer = if let Some(ref w) = self.weights {
            w.buffer()
        } else {
            sentinel_buf = device.create_buffer_f32(1)?;
            &sentinel_buf
        };

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            has_weights: u32,
            num_weights: u32,
            num_partials: u32,
        }

        let params = Params {
            size: size as u32,
            has_weights,
            num_weights,
            num_partials: num_workgroups,
        };
        let params_buffer = device.create_uniform_buffer("PerceptualLoss Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let workgroups = caps.dispatch_1d(size as u32);

        ComputeDispatch::new(device, "perceptual_loss_pass1")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.features1.buffer())
            .storage_read(1, self.features2.buffer())
            .storage_read(2, weights_buffer)
            .storage_rw(3, &loss_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        ComputeDispatch::new(device, "perceptual_loss_pass2")
            .shader(Self::wgsl_shader(), "compute_mean_loss")
            .storage_read(0, self.features1.buffer())
            .storage_read(1, self.features2.buffer())
            .storage_read(2, weights_buffer)
            .storage_rw(3, &loss_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(1, 1, 1)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &output_buffer, 1)?;
        Ok(Tensor::new(output_data, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_perceptual_loss() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let features1 = Tensor::from_vec_on(vec![0.5; 1000], vec![1000], device.clone())
            .await
            .unwrap();
        let features2 = Tensor::from_vec_on(vec![0.6; 1000], vec![1000], device.clone())
            .await
            .unwrap();
        let loss = PerceptualLoss::new(features1, features2, None)
            .unwrap()
            .execute()
            .unwrap();
        let result = loss.to_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0] > 0.0);
    }
}
