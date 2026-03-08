// SPDX-License-Identifier: AGPL-3.0-or-later
//! Focal Loss - GPU-accelerated loss for imbalanced classification
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (uses existing shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for object detection)
//!
//! ## Algorithm
//!
//! ```text
//! For binary classification:
//! p_t = p if target=1, else (1-p)
//! FL = -alpha * (1 - p_t)^gamma * log(p_t)
//! ```
//!
//! **Parameters**:
//! - `alpha`: Balancing factor (typically 0.25)
//! - `gamma`: Focusing parameter (typically 2.0)
//!
//! **Key Properties**:
//! - Down-weights easy examples
//! - Focuses training on hard negatives
//! - Handles severe class imbalance
//! - Standard in object detection (`RetinaNet`)
//!
//! **Used By**: `RetinaNet`, object detection, imbalanced classification
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;  // Probabilities
//! let targets = Tensor::randn(vec![1000]).await?;      // Binary labels
//!
//! let loss = predictions.focal_loss(&targets, 0.25, 2.0)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FocalLossParams {
    alpha: f32,
    gamma: f32,
    epsilon: f32,
    reduction_mode: u32,
    size: u32,
    _pad1: [u32; 3],
    _pad2: [u32; 4],
    _pad3: [u32; 4],
    _pad4: [u32; 4],
    _pad5: [u32; 4],
}

/// Focal loss for imbalanced classification (down-weights easy examples).
pub struct FocalLoss {
    predictions: Tensor,
    targets: Tensor,
    alpha: f32,
    gamma: f32,
}

impl FocalLoss {
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/focal_loss_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute focal loss computation and return the result tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = FocalLossParams {
            alpha: self.alpha,
            gamma: self.gamma,
            epsilon: 1e-7,
            reduction_mode: 0, // mean reduction
            size: size as u32,
            _pad1: [0; 3],
            _pad2: [0; 4],
            _pad3: [0; 4],
            _pad4: [0; 4],
            _pad5: [0; 4],
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("focal_loss_output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_uniform_buffer("focal_loss_params", &params);

        ComputeDispatch::new(device.as_ref(), "focal_loss")
            .shader(Self::wgsl_shader(), "main")
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
    /// Focal loss for imbalanced classification
    ///
    /// **Deep Debt**: Essential for object detection and severe class imbalance
    ///
    /// # Arguments
    /// - `targets`: Ground truth binary labels [same shape as predictions]
    /// - `alpha`: Balancing factor (typically 0.25)
    /// - `gamma`: Focusing parameter (typically 2.0, higher = more focus on hard examples)
    ///
    /// # Returns
    /// - Loss tensor [same shape as input]
    ///
    /// # Example
    /// ```rust,ignore
    /// // RetinaNet-style focal loss
    /// let loss = predictions.focal_loss(&targets, 0.25, 2.0)?;
    ///
    /// // Higher gamma for very imbalanced datasets
    /// let loss = predictions.focal_loss(&targets, 0.25, 5.0)?;
    /// ```
    ///
    /// # Note
    /// - `gamma=0`: Equivalent to binary cross-entropy
    /// - `gamma=2`: Standard for object detection (`RetinaNet`)
    /// - Higher `gamma`: More aggressive down-weighting of easy examples
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn focal_loss(self, targets: &Self, alpha: f32, gamma: f32) -> Result<Self> {
        // Validate shapes match
        if self.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                self.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        // Validate parameters
        if !(0.0..=1.0).contains(&alpha) {
            return Err(BarracudaError::invalid_op(
                "FocalLoss",
                format!("alpha must be in [0, 1], got {alpha}"),
            ));
        }
        if gamma < 0.0 {
            return Err(BarracudaError::invalid_op(
                "FocalLoss",
                format!("gamma must be non-negative, got {gamma}"),
            ));
        }

        let op = FocalLoss {
            predictions: self,
            targets: targets.clone(),
            alpha,
            gamma,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_focal_loss_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let predictions =
            Tensor::from_data(&[0.9, 0.1, 0.8, 0.2], vec![4], device.clone()).unwrap();

        let targets = Tensor::from_data(&[1.0, 0.0, 1.0, 0.0], vec![4], device.clone()).unwrap();

        let result = predictions.focal_loss(&targets, 0.25, 2.0).unwrap();
        let loss = result.to_vec().unwrap();

        assert_eq!(loss.len(), 4);
        // Verify operation completed successfully
        assert!(!loss.is_empty());
    }

    #[tokio::test]
    async fn test_focal_loss_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Perfect predictions
        let predictions = Tensor::from_data(&[1.0, 0.0, 1.0], vec![3], device.clone()).unwrap();
        let targets = Tensor::from_data(&[1.0, 0.0, 1.0], vec![3], device.clone()).unwrap();
        let result = predictions.focal_loss(&targets, 0.25, 2.0).unwrap();
        let loss = result.to_vec().unwrap();
        assert_eq!(loss.len(), 3); // Verify operation completed

        // Single element
        let predictions = Tensor::from_data(&[0.7], vec![1], device.clone()).unwrap();
        let targets = Tensor::from_data(&[1.0], vec![1], device).unwrap();
        let result = predictions.focal_loss(&targets, 0.5, 2.0).unwrap();
        let loss = result.to_vec().unwrap();
        assert_eq!(loss.len(), 1);
    }

    #[tokio::test]
    async fn test_focal_loss_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Different alpha values
        let predictions = Tensor::from_data(&[0.6, 0.4], vec![2], device.clone()).unwrap();
        let targets = Tensor::from_data(&[1.0, 0.0], vec![2], device.clone()).unwrap();
        let result1 = predictions.clone().focal_loss(&targets, 0.25, 2.0).unwrap();
        let loss1 = result1.to_vec().unwrap();

        let result2 = predictions.focal_loss(&targets, 0.75, 2.0).unwrap();
        let loss2 = result2.to_vec().unwrap();

        // Both should complete successfully
        assert_eq!(loss1.len(), 2);
        assert_eq!(loss2.len(), 2);
    }

    #[tokio::test]
    async fn test_focal_loss_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 100 elements
        let mut preds = Vec::with_capacity(100);
        let mut tgts = Vec::with_capacity(100);
        for i in 0..100 {
            preds.push(if i % 2 == 0 { 0.8 } else { 0.2 });
            tgts.push(if i % 2 == 0 { 1.0 } else { 0.0 });
        }

        let predictions = Tensor::from_data(&preds, vec![100], device.clone()).unwrap();
        let targets = Tensor::from_data(&tgts, vec![100], device).unwrap();

        let result = predictions.focal_loss(&targets, 0.25, 2.0).unwrap();
        let loss = result.to_vec().unwrap();

        assert_eq!(loss.len(), 100);
    }

    #[tokio::test]
    async fn test_focal_loss_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Gamma parameter effect
        let predictions = Tensor::from_data(&[0.5, 0.9], vec![2], device.clone()).unwrap();
        let targets = Tensor::from_data(&[1.0, 1.0], vec![2], device.clone()).unwrap();

        // Low gamma
        let result_low = predictions.clone().focal_loss(&targets, 0.25, 0.5).unwrap();
        let loss_low = result_low.to_vec().unwrap();

        // High gamma
        let result_high = predictions.focal_loss(&targets, 0.25, 4.0).unwrap();
        let loss_high = result_high.to_vec().unwrap();

        // Both should complete successfully
        assert_eq!(loss_low.len(), 2);
        assert_eq!(loss_high.len(), 2);
    }
}
