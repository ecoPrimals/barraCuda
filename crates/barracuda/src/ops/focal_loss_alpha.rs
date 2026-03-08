// SPDX-License-Identifier: AGPL-3.0-or-later
//! Focal Loss with class weighting
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Extension of focal loss with per-class weights (alpha)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FocalLossAlphaParams {
    batch_size: u32,
    num_classes: u32,
    gamma: f32,
    _padding: u32,
}

/// Focal loss with per-class alpha weighting for imbalanced classification.
pub struct FocalLossAlpha {
    predictions: Tensor,
    targets: Tensor,
    alpha: Tensor,
    gamma: f32,
}

impl FocalLossAlpha {
    /// Create `FocalLossAlpha` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(predictions: Tensor, targets: Tensor, alpha: Tensor, gamma: f32) -> Result<Self> {
        if gamma < 0.0 {
            return Err(BarracudaError::invalid_op(
                "FocalLossAlpha",
                format!("gamma must be non-negative, got {gamma}"),
            ));
        }

        Ok(Self {
            predictions,
            targets,
            alpha,
            gamma,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/focal_loss_alpha_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute `FocalLossAlpha` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let pred_shape = self.predictions.shape();
        let target_shape = self.targets.shape();

        if pred_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "FocalLossAlpha",
                format!("predictions must be 2D [batch, num_classes], got shape {pred_shape:?}"),
            ));
        }

        if target_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "FocalLossAlpha",
                format!("targets must be 1D [batch], got shape {target_shape:?}"),
            ));
        }

        let batch_size = pred_shape[0];
        let num_classes = pred_shape[1];

        if target_shape[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "FocalLossAlpha",
                format!(
                    "targets batch size {} must match predictions batch size {}",
                    target_shape[0], batch_size
                ),
            ));
        }

        if self.alpha.shape() != [num_classes] {
            return Err(BarracudaError::invalid_op(
                "FocalLossAlpha",
                format!(
                    "alpha must be 1D [num_classes], got shape {:?}",
                    self.alpha.shape()
                ),
            ));
        }

        // Create output buffer: [batch] - per-sample loss
        let output_buffer = device.create_buffer_f32(batch_size)?;

        let params = FocalLossAlphaParams {
            batch_size: batch_size as u32,
            num_classes: num_classes as u32,
            gamma: self.gamma,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("FocalLossAlpha Params", &params);

        ComputeDispatch::new(device, "FocalLossAlpha")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_read(2, self.alpha.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_focal_loss_alpha_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 4;
        let num_classes = 3;

        let predictions = Tensor::from_vec_on(
            vec![0.33; batch_size * num_classes],
            vec![batch_size, num_classes],
            device.clone(),
        )
        .await
        .unwrap();

        let targets =
            Tensor::from_vec_on(vec![0.0, 1.0, 2.0, 0.0], vec![batch_size], device.clone())
                .await
                .unwrap();

        let alpha = Tensor::from_vec_on(vec![0.25, 0.25, 0.5], vec![num_classes], device.clone())
            .await
            .unwrap();

        let result = FocalLossAlpha::new(predictions, targets, alpha, 2.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size]);
    }
}
