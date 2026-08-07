// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dice Loss - Medical image segmentation loss function
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper
//! - ✅ Handles class imbalance (medical imaging standard)
//!
//! ## Algorithm
//!
//! Dice Loss = 1 - Dice Coefficient
//! Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
//!
//! Where X = predicted, Y = target, smooth prevents division by zero
//!
//! ## Usage
//!
//! ```rust,ignore
//! let predicted = Tensor::sigmoid(logits)?; // [0, 1] probabilities
//! let target = Tensor::from_vec(ground_truth, shape).await?;
//! let loss = predicted.dice_loss(&target, 1.0)?; // smooth = 1.0
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DiceLossParams {
    size: u32,
    smooth_val: f32,
    _padding: [u32; 2],
}

/// Dice loss for medical image segmentation (handles class imbalance).
pub struct DiceLoss {
    predicted: Tensor,
    target: Tensor,
    smooth: f32,
}

impl DiceLoss {
    /// Creates a new Dice loss. Smooth prevents division by zero.
    /// # Errors
    /// Returns [`Err`] if predicted and target shapes do not match.
    pub fn new(predicted: Tensor, target: Tensor, smooth: f32) -> Result<Self> {
        if predicted.shape() != target.shape() {
            return Err(BarracudaError::shape_mismatch(
                predicted.shape().to_vec(),
                target.shape().to_vec(),
            ));
        }

        Ok(Self {
            predicted,
            target,
            smooth,
        })
    }

    /// Executes Dice loss and returns a scalar loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predicted.device();
        let size = self.predicted.len();

        // Output is scalar loss value
        let output_buffer = device.create_buffer_f32(1)?;

        let params = DiceLossParams {
            size: size as u32,
            smooth_val: self.smooth,
            _padding: [0, 0],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dice Loss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Dice Loss")
            .shader(include_str!("../shaders/loss/dice_loss_f64.wgsl"), "main")
            .storage_read(0, self.predicted.buffer())
            .storage_read(1, self.target.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dice_loss_perfect_overlap() {
        let device = crate::device::test_pool::get_test_device().await;
        // Perfect prediction = target
        let pred = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // Perfect overlap: Dice = 1.0, Loss = 0.0
        assert!(result[0] < 0.1, "Perfect overlap should have loss ≈ 0");
    }

    #[tokio::test]
    async fn test_dice_loss_no_overlap() {
        let device = crate::device::test_pool::get_test_device().await;
        // No overlap
        let pred = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![0.0, 0.0, 1.0, 1.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // No overlap: Loss should be high (close to 1.0)
        assert!(result[0] > 0.5, "No overlap should have high loss");
    }

    #[tokio::test]
    async fn test_dice_loss_partial_overlap() {
        let device = crate::device::test_pool::get_test_device().await;
        let pred = Tensor::from_vec_on(vec![0.8, 0.6, 0.2, 0.1], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // Partial overlap: 0 < Loss < 1
        assert!(result[0] > 0.0 && result[0] < 1.0);
        assert!(result[0].is_finite());
    }
}
