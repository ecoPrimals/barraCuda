// SPDX-License-Identifier: AGPL-3.0-or-later
//! Huber Loss - GPU-accelerated robust regression loss
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (uses existing shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for robust regression)
//!
//! ## Algorithm
//!
//! ```text
//! Huber(x, y, δ) = 0.5 * (x - y)²             if |x - y| ≤ δ
//!                = δ * (|x - y| - 0.5 * δ)    otherwise
//! ```
//!
//! **Parameters**:
//! - `delta`: Threshold for switching from quadratic (MSE) to linear (MAE)
//!
//! **Key Properties**:
//! - Robust to outliers (less sensitive than MSE)
//! - Smooth at zero (differentiable everywhere)
//! - Quadratic for small errors, linear for large errors
//! - Standard in reinforcement learning (DQN)
//!
//! **Used By**: DQN, robust regression, outlier-resistant training
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;
//! let targets = Tensor::randn(vec![1000]).await?;
//!
//! let loss = predictions.huber_loss(&targets, 1.0)?;  // delta=1.0
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/loss/huber_loss_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HuberLossParams {
    delta: f32,
    reduction_mode: u32,
    size: u32,
    _padding: u32,
}

pub struct HuberLoss {
    predictions: Tensor,
    targets: Tensor,
    delta: f32,
}

impl HuberLoss {
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = HuberLossParams {
            delta: self.delta,
            reduction_mode: 0, // mean reduction
            size: size as u32,
            _padding: 0,
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("huber_loss_output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_uniform_buffer("huber_loss_params", &params);

        ComputeDispatch::new(device, "huber_loss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit();

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
    /// Huber loss for robust regression (less sensitive to outliers than MSE)
    ///
    /// **Deep Debt**: Essential for robust regression and reinforcement learning
    ///
    /// # Arguments
    /// - `targets`: Ground truth values [same shape as predictions]
    /// - `delta`: Threshold for switching from quadratic to linear (typically 1.0)
    ///
    /// # Returns
    /// - Loss tensor [same shape as input]
    ///
    /// # Example
    /// ```rust,ignore
    /// // DQN-style Huber loss
    /// let loss = predictions.huber_loss(&targets, 1.0)?;
    ///
    /// // More sensitive to outliers (larger delta)
    /// let loss = predictions.huber_loss(&targets, 2.0)?;
    /// ```
    ///
    /// # Note
    /// - `delta=1.0`: Standard for DQN (Deep Q-Network)
    /// - Small delta: More robust (less outlier influence)
    /// - Large delta: Closer to MSE behavior
    /// - Combines MSE (small errors) + MAE (large errors)
    pub fn huber_loss(self, targets: &Self, delta: f32) -> Result<Self> {
        // Validate shapes match
        if self.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                self.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        // Validate delta is positive
        if delta <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "HuberLoss",
                format!("delta must be positive, got {delta}"),
            ));
        }

        let op = HuberLoss {
            predictions: self,
            targets: targets.clone(),
            delta,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_huber_loss_small_errors() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Small errors (< delta): should use quadratic (MSE-like)
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![1.1, 2.1, 2.9, 3.9], vec![4], device.clone())
            .await
            .unwrap();

        let result = predictions.huber_loss(&targets, 1.0).unwrap();
        let loss = result.to_vec().unwrap();

        assert_eq!(loss.len(), 4);
        // All errors = 0.1, which is < delta=1.0
        // Loss should be 0.5 * 0.1^2 = 0.005
        for &l in &loss {
            assert!((l - 0.005).abs() < 1e-5, "Expected 0.005, got {}", l);
        }
    }

    #[tokio::test]
    async fn test_huber_loss_large_errors() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Large errors (> delta): should use linear (MAE-like)
        let predictions = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
            .await
            .unwrap();

        let targets = Tensor::from_vec_on(vec![3.0, 5.0], vec![2], device.clone())
            .await
            .unwrap();

        let result = predictions.huber_loss(&targets, 1.0).unwrap();
        let loss = result.to_vec().unwrap();

        assert_eq!(loss.len(), 2);
        // Error 1: |1-3| = 2 > delta=1, loss = 1*(2 - 0.5*1) = 1.5
        assert!(
            (loss[0] - 1.5).abs() < 1e-5,
            "Expected 1.5, got {}",
            loss[0]
        );
        // Error 2: |2-5| = 3 > delta=1, loss = 1*(3 - 0.5*1) = 2.5
        assert!(
            (loss[1] - 2.5).abs() < 1e-5,
            "Expected 2.5, got {}",
            loss[1]
        );
    }
}
