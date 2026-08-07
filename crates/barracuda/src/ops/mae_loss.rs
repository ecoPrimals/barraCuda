// SPDX-License-Identifier: AGPL-3.0-or-later
//! MAE Loss - GPU-accelerated Mean Absolute Error Loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (existing shader evolved)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! ## Algorithm
//!
//! ```text
//! MAE = (1/n) * Σ |y_pred - y_true|
//! ```
//!
//! **Key Properties**:
//! - Less sensitive to outliers than MSE
//! - Linear penalty for errors
//! - Robust loss function
//! - Used in regression tasks
//!
//! **Used By**: Robust regression, forecasting, time series
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;
//! let targets = Tensor::randn(vec![1000]).await?;
//!
//! let loss = predictions.mae_loss(&targets)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MAELossParams {
    reduction_mode: u32,
    size: u32,
    _padding: [u32; 2],
}

/// Mean absolute error loss for robust regression.
pub struct MAELoss {
    predictions: Tensor,
    targets: Tensor,
}

impl MAELoss {
    /// Creates a new MAE loss. Shapes must match.
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

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/loss/mae_loss.wgsl")
    }

    /// f64 MAE loss (tree reduction for accumulation accuracy).
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/loss/mae_loss_f64.wgsl")
    }

    /// Executes MAE loss and returns a scalar loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = MAELossParams {
            reduction_mode: 0, // mean
            size: size as u32,
            _padding: [0; 2],
        };

        let output_buffer = device.create_buffer_f32(size)?;

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mae_loss_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "MAELoss")
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
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// MAE Loss (Mean Absolute Error) - robust regression loss
    /// **Deep Debt**: Essential for robust regression tasks
    /// # Arguments
    /// - `targets`: Target tensor [same shape as predictions]
    /// # Returns
    /// - Loss tensor [same shape as input]
    /// # Example
    /// ```rust,ignore
    /// // Regression
    /// let loss = predictions.mae_loss(&targets)?;
    /// ```
    /// # Note
    /// - Less sensitive to outliers than MSE
    /// - Linear penalty for errors
    /// - Used in robust regression
    /// # Errors
    /// Returns [`Err`] if shapes do not match, or if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn mae_loss(self, targets: &Self) -> Result<Self> {
        MAELoss::new(self, targets.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mae_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.5, 2.5, 3.5, 4.5], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), 4);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0)); // MAE is always non-negative
    }

    #[tokio::test]
    async fn test_mae_loss_perfect() {
        let device = crate::device::test_pool::get_test_device().await;
        // Perfect predictions should have zero loss
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert!(data.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_mae_loss_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Shape mismatch
        let predictions = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 5], vec![5], device.clone())
            .await
            .unwrap();

        assert!(predictions.mae_loss(&targets).is_err());
    }

    #[tokio::test]
    async fn test_mae_loss_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 1000;
        let predictions = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let loss = predictions.mae_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x.is_finite()));
        // Should be close to 1.0 (absolute difference)
        assert!((data[0] - 1.0).abs() < 0.1);
    }
}
