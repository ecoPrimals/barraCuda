// SPDX-License-Identifier: AGPL-3.0-or-later
//! L1 Loss - Mean Absolute Error - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Computes Mean Absolute Error loss:
//! ```text
//! L1(pred, target) = mean(|pred - target|)
//!
//! Element-wise: loss[i] = |pred[i] - target[i]|
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Mean Absolute Error (L1) loss.
pub struct L1Loss {
    predictions: Tensor,
    targets: Tensor,
}

impl L1Loss {
    /// Create an L1 loss operation between predictions and targets.
    #[must_use]
    pub fn new(predictions: Tensor, targets: Tensor) -> Self {
        Self {
            predictions,
            targets,
        }
    }

    /// Execute L1 loss computation on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.len();

        if size != self.targets.len() {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: self.predictions.shape().to_vec(),
                actual: self.targets.shape().to_vec(),
            });
        }

        let output_buffer = device.create_buffer_f32(size)?;

        // Create params buffer (size and reduction mode)
        let params_data = [
            size as u32,
            0u32, // reduction: 0=none (element-wise)
        ];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        ComputeDispatch::new(device, "L1Loss")
            .shader(include_str!("../shaders/loss/l1_loss_f64.wgsl"), "main")
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

impl Tensor {
    /// Compute element-wise L1 loss (absolute error) against targets.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn l1_loss_wgsl(self, targets: Self) -> Result<Self> {
        L1Loss::new(self, targets).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_l1_loss() {
        let device = crate::device::test_pool::get_test_device().await;
        let pred_data = vec![1.0, 2.0, 3.0, 4.0];
        let target_data = vec![1.5, 2.5, 2.0, 5.0];

        let predictions = Tensor::from_vec_on(pred_data, vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(target_data, vec![4], device)
            .await
            .unwrap();

        let result = predictions.l1_loss_wgsl(targets).unwrap();
        let output = result.to_vec().unwrap();

        // |1.0 - 1.5| = 0.5
        assert!((output[0] - 0.5).abs() < 1e-5);
        // |2.0 - 2.5| = 0.5
        assert!((output[1] - 0.5).abs() < 1e-5);
        // |3.0 - 2.0| = 1.0
        assert!((output[2] - 1.0).abs() < 1e-5);
        // |4.0 - 5.0| = 1.0
        assert!((output[3] - 1.0).abs() < 1e-5);
    }
}
