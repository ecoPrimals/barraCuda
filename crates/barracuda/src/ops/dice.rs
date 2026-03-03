// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dice Loss - GPU-accelerated segmentation loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (uses existing shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for medical imaging)
//!
//! ## Algorithm
//!
//! ```text
//! Dice coefficient: DC = 2|X ∩ Y| / (|X| + |Y|)
//! Dice loss: L = 1 - DC
//! ```
//!
//! **Implementation**: GPU reduction with workgroup shared memory
//!
//! **Key Properties**:
//! - Handles class imbalance naturally
//! - Directly optimizes IoU-like metric
//! - Common in medical image segmentation
//!
//! **Used By**: U-Net, V-Net, medical imaging models
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![4, 256, 256]).await?;  // [batch, H, W]
//! let targets = Tensor::randn(vec![4, 256, 256]).await?;
//!
//! let loss = predictions.dice_loss(&targets, 1.0)?;  // smoothing = 1.0
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Dice loss parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DiceParams {
    smoothing: f32,
    reduction_mode: u32,
    batch_size: u32,
    elements_per_sample: u32,
}

/// Dice Loss operation
///
/// **Deep Debt**: Uses existing WGSL shader with workgroup reduction
pub struct DiceLoss {
    predictions: Tensor,
    targets: Tensor,
    smoothing: f32,
}

impl DiceLoss {
    /// Create new Dice loss operation
    pub fn new(predictions: Tensor, targets: Tensor, smoothing: f32) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        Ok(Self {
            predictions,
            targets,
            smoothing,
        })
    }

    /// WGSL shader source
    fn shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/dice_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute Dice loss (GPU reduction)
    ///
    /// **Deep Debt**: Efficient workgroup reduction for large segmentations
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();

        // Determine batch and elements
        let total_size = self.predictions.len();
        let batch_size = if !self.predictions.shape().is_empty() {
            self.predictions.shape()[0]
        } else {
            1
        };
        let elements_per_sample = total_size / batch_size;

        // Create parameters
        let params = DiceParams {
            smoothing: self.smoothing,
            reduction_mode: 0, // Mean reduction
            batch_size: batch_size as u32,
            elements_per_sample: elements_per_sample as u32,
        };

        let params_buffer = device.create_uniform_buffer("Dice Loss Params", &params);

        // Output buffer (one loss value per batch)
        let output_buffer = device.create_buffer_f32(batch_size)?;

        ComputeDispatch::new(device, "Dice Loss")
            .shader(Self::shader(), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit();

        // Return output tensor [batch_size]
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Dice loss for segmentation
    ///
    /// **Deep Debt**: Essential for medical imaging, handles class imbalance
    ///
    /// # Arguments
    /// - `targets`: Ground truth [same shape as predictions]
    /// - `smoothing`: Smoothing factor (typically 1.0)
    ///
    /// # Returns
    /// - Loss tensor [batch_size] (one value per batch)
    ///
    /// # Example
    /// ```rust,ignore
    /// let preds = Tensor::randn(vec![4, 256, 256]).await?;
    /// let targets = Tensor::randn(vec![4, 256, 256]).await?;
    /// let loss = preds.dice_loss(&targets, 1.0)?;  // U-Net segmentation
    /// ```
    pub fn dice_loss(self, targets: &Self, smoothing: f32) -> Result<Self> {
        DiceLoss::new(self, targets.clone(), smoothing)?.execute()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_dice_loss_gpu_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 2;
        let h = 16;
        let w = 16;

        // Perfect predictions
        let preds =
            Tensor::from_vec_on(vec![1.0; batch * h * w], vec![batch, h, w], device.clone())
                .await
                .unwrap();

        let targets = Tensor::from_vec_on(vec![1.0; batch * h * w], vec![batch, h, w], device)
            .await
            .unwrap();

        let loss = preds.dice_loss(&targets, 1.0).unwrap();

        assert_eq!(loss.shape(), &[batch]);
        let data = loss.to_vec().unwrap();
        // Perfect prediction should have low loss
        assert!(data.iter().all(|&x| x < 0.1));
    }

    #[tokio::test]
    async fn test_dice_loss_gpu_mismatch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let size = 100;

        // Complete mismatch
        let preds = Tensor::from_vec_on(vec![1.0; batch * size], vec![batch, size], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![0.0; batch * size], vec![batch, size], device)
            .await
            .unwrap();

        let loss = preds.dice_loss(&targets, 1.0).unwrap();
        let data = loss.to_vec().unwrap();

        // High loss for mismatch
        assert!(data[0] > 0.5);
    }

    #[tokio::test]
    async fn test_dice_loss_gpu_medical_scale() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Medical imaging scale: 128x128 slices
        let batch = 4;
        let h = 128;
        let w = 128;

        let preds =
            Tensor::from_vec_on(vec![0.8; batch * h * w], vec![batch, h, w], device.clone())
                .await
                .unwrap();

        let targets = Tensor::from_vec_on(vec![1.0; batch * h * w], vec![batch, h, w], device)
            .await
            .unwrap();

        let loss = preds.dice_loss(&targets, 1.0).unwrap();

        assert_eq!(loss.shape(), &[batch]);
        let data = loss.to_vec().unwrap();
        assert!(data
            .iter()
            .all(|&x| x.is_finite() && (0.0..=1.0).contains(&x)));
    }
}
