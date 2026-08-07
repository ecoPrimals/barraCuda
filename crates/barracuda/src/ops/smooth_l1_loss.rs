// SPDX-License-Identifier: AGPL-3.0-or-later
//! Smooth L1 Loss - GPU-accelerated Robust Regression Loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (NEW shader created!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (Tensor API)
//!
//! ## Algorithm
//!
//! ```text
//! diff = |pred - target|
//! if diff < beta:
//!     loss = 0.5 * diff² / beta
//! else:
//!     loss = diff - 0.5 * beta
//! ```
//!
//! **Key Properties**:
//! - Combines L1 and L2 loss benefits
//! - Quadratic below beta (smooth near zero)
//! - Linear above beta (robust to outliers)
//! - Less sensitive to outliers than MSE
//! - Smooth gradient transition at beta
//!
//! **Parameters**:
//! - `beta`: Threshold between quadratic and linear regions, typically 1.0
//!
//! **Used By**: Object detection (Faster R-CNN), robust regression
//!
//! **Related**: Similar to Huber Loss (different parameterization)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;
//! let targets = Tensor::randn(vec![1000]).await?;
//!
//! let loss = predictions.smooth_l1_loss(&targets, 1.0)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/loss/smooth_l1_loss_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SmoothL1LossParams {
    beta: f32,
    size: u32,
    _padding: [u32; 2],
}

/// Smooth L1 (Huber-like) loss for robust regression.
pub struct SmoothL1Loss {
    predictions: Tensor,
    targets: Tensor,
    beta: f32,
}

impl SmoothL1Loss {
    /// Creates a new smooth L1 loss. `beta` is the threshold between quadratic and linear regions.
    /// # Errors
    /// Returns [`Err`] if prediction and target shapes do not match, or beta is not positive.
    pub fn new(predictions: Tensor, targets: Tensor, beta: f32) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        // Validate beta is positive
        if beta <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "smooth_l1_loss",
                "beta must be positive",
            ));
        }

        Ok(Self {
            predictions,
            targets,
            beta,
        })
    }

    /// Executes the smooth L1 loss computation and returns the loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = SmoothL1LossParams {
            beta: self.beta,
            size: size as u32,
            _padding: [0; 2],
        };

        let output_buffer = device.create_buffer_f32(size)?;

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("smooth_l1_loss_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "SmoothL1Loss")
            .shader(SHADER_F64, "main")
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
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Smooth L1 Loss - robust regression loss for object detection
    /// **Deep Debt**: Essential for Faster R-CNN and robust regression
    /// # Arguments
    /// - `targets`: Target tensor [same shape as predictions]
    /// - `beta`: Threshold between quadratic/linear regions, typically 1.0
    /// # Returns
    /// - Loss tensor [same shape as input]
    /// # Example
    /// ```rust,ignore
    /// // Object detection bounding box regression
    /// let loss = bbox_pred.smooth_l1_loss(&bbox_target, 1.0)?;
    /// ```
    /// # Note
    /// - Combines L1 and L2 benefits
    /// - Quadratic below beta (smooth)
    /// - Linear above beta (robust to outliers)
    /// - beta must be positive
    /// - Similar to Huber Loss (different parameterization)
    /// # Errors
    /// Returns [`Err`] if shapes do not match, beta is not positive, or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn smooth_l1_loss(self, targets: &Self, beta: f32) -> Result<Self> {
        SmoothL1Loss::new(self, targets.clone(), beta)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_smooth_l1_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.5, 2.5, 3.5], vec![3], device.clone())
            .await
            .unwrap();

        let loss = predictions.smooth_l1_loss(&targets, 1.0).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), 3);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0)); // Loss is always non-negative
    }

    #[tokio::test]
    async fn test_smooth_l1_loss_perfect() {
        let device = crate::device::test_pool::get_test_device().await;
        // Perfect predictions should have zero loss
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let loss = predictions.smooth_l1_loss(&targets, 1.0).unwrap();
        let data = loss.to_vec().unwrap();

        assert!(data.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_smooth_l1_loss_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test behavior at beta boundary
        let beta = 1.0;

        // Small diff (quadratic region)
        let predictions = Tensor::from_vec_on(vec![0.0], vec![1], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![0.5], vec![1], device.clone())
            .await
            .unwrap();

        let loss_small = predictions.smooth_l1_loss(&targets, beta).unwrap();
        let data_small = loss_small.to_vec().unwrap();

        // Large diff (linear region)
        let predictions = Tensor::from_vec_on(vec![0.0], vec![1], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![2.0], vec![1], device.clone())
            .await
            .unwrap();

        let loss_large = predictions.smooth_l1_loss(&targets, beta).unwrap();
        let data_large = loss_large.to_vec().unwrap();

        assert!(data_small[0] < data_large[0]);
        assert!(data_small[0] < 0.5); // Quadratic region
        assert!(data_large[0] > 1.0); // Linear region
    }

    #[tokio::test]
    async fn test_smooth_l1_loss_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Shape mismatch
        let predictions = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 5], vec![5], device.clone())
            .await
            .unwrap();

        assert!(predictions.clone().smooth_l1_loss(&targets, 1.0).is_err());

        // Invalid beta
        let targets_correct = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();

        assert!(
            predictions
                .clone()
                .smooth_l1_loss(&targets_correct, -1.0)
                .is_err()
        );
        assert!(predictions.smooth_l1_loss(&targets_correct, 0.0).is_err());
    }

    #[tokio::test]
    async fn test_smooth_l1_loss_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 1000;
        let predictions = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let loss = predictions.smooth_l1_loss(&targets, 1.0).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_smooth_l1_loss_different_betas() {
        let device = crate::device::test_pool::get_test_device().await;
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![2.0, 3.0], vec![2], device.clone())
            .await
            .unwrap();

        let loss_beta1 = predictions
            .clone()
            .smooth_l1_loss(&targets, 1.0)
            .unwrap()
            .to_vec()
            .unwrap();

        let loss_beta2 = predictions
            .smooth_l1_loss(&targets, 2.0)
            .unwrap()
            .to_vec()
            .unwrap();

        // Different betas should produce different losses
        assert!(loss_beta1[0] != loss_beta2[0]);
    }
}
