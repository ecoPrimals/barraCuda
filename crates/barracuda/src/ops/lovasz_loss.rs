// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lovasz Loss - GPU-accelerated IoU-optimized loss for segmentation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (new shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for segmentation)
//!
//! ## Algorithm
//!
//! ```text
//! Lovasz Loss = Lovasz_extension(IoU_loss)
//!
//! Steps:
//! 1. Compute errors: e = max(0, 1 - p_true)
//! 2. Sort errors in descending order
//! 3. Compute Lovasz extension for IoU
//!
//! Benefits: Directly optimizes IoU metric
//! ```
//!
//! **Key Properties**:
//! - Convex surrogate for `IoU` loss
//! - Directly optimizes Intersection over Union
//! - Better than cross-entropy for segmentation
//! - Especially effective for imbalanced classes
//!
//! **Used By**: Semantic segmentation, medical imaging, scene understanding
//!
//! **Reference**: "The Lovász-Softmax loss" (Berman et al., CVPR 2018)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;  // Predicted probs
//! let targets = Tensor::randn(vec![1000]).await?;      // Ground truth
//!
//! let loss = predictions.lovasz_loss(&targets)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LovaszLossParams {
    size: u32,
    smooth: f32,
    _padding: [u32; 2],
}

/// Lovász-Softmax loss for IoU-optimized segmentation.
pub struct LovaszLoss {
    predictions: Tensor,
    targets: Tensor,
}

impl LovaszLoss {
    /// Creates a new Lovász loss. Shapes must match.
    /// # Errors
    /// Returns [`Err`] if prediction and target shapes do not match.
    pub fn new(predictions: Tensor, targets: Tensor) -> Result<Self> {
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
        })
    }

    /// Executes Lovász loss and returns the loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = LovaszLossParams {
            size: size as u32,
            smooth: 1e-5,
            _padding: [0; 2],
        };

        let output_buffer = device.create_buffer_f32(size)?;

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lovasz_loss_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "LovaszLoss")
            .shader(include_str!("../shaders/loss/lovasz_loss_f64.wgsl"), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.predictions.shape().to_vec(),
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Lovasz Loss for IoU-optimized semantic segmentation
    /// **Deep Debt**: Essential for semantic segmentation tasks
    /// # Arguments
    /// - `targets`: Ground truth tensor [same shape as predictions]
    /// # Returns
    /// - Loss tensor [same shape as input]
    /// # Example
    /// ```rust,ignore
    /// // Semantic segmentation
    /// let loss = predictions.lovasz_loss(&targets)?;
    /// // Medical imaging
    /// let seg_loss = model_output.lovasz_loss(&ground_truth)?;
    /// ```
    /// # Note
    /// - Directly optimizes `IoU` metric
    /// - Better than cross-entropy for segmentation
    /// - Especially effective for imbalanced classes
    /// - Predictions and targets should be in [0, 1]
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn lovasz_loss(self, targets: &Self) -> Result<Self> {
        LovaszLoss::new(self, targets.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lovasz_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let predictions = Tensor::from_vec_on(vec![0.9, 0.8, 0.7, 0.6], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), 4);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0)); // Loss should be non-negative
    }

    #[tokio::test]
    async fn test_lovasz_loss_perfect_prediction() {
        let device = crate::device::test_pool::get_test_device().await;
        // Perfect prediction should have very low loss
        let predictions = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        assert!(
            mean < 0.1,
            "Expected low loss for perfect prediction, got {mean}"
        );
    }

    #[tokio::test]
    async fn test_lovasz_loss_poor_prediction() {
        let device = crate::device::test_pool::get_test_device().await;
        // Poor prediction should have higher loss
        let predictions = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.1], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert!(data.iter().all(|&x| x > 0.5)); // Should have high error
    }

    #[tokio::test]
    async fn test_lovasz_loss_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Shape mismatch
        let predictions = Tensor::from_vec_on(vec![0.5; 10], vec![10], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 5], vec![5], device.clone())
            .await
            .unwrap();

        assert!(predictions.lovasz_loss(&targets).is_err());
    }

    #[tokio::test]
    async fn test_lovasz_loss_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 1000;
        let pred_data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();
        let target_data = vec![1.0; size];

        let predictions = Tensor::from_vec_on(pred_data, vec![size], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(target_data, vec![size], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0));
    }
}
