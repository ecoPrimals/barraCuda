// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leave-One-Out Cross-Validation for kernel methods - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Leave-one-out cross-validation residuals for kernel methods.
/// `LOO_i` = (`y_i` - `pred_i`) / (1 - `H_ii`)
pub struct LooCv {
    hat_matrix: Tensor,
    y: Tensor,
    predictions: Tensor,
}

impl LooCv {
    /// Create LOO-CV residuals from hat matrix, targets, and predictions.
    #[must_use]
    pub fn new(hat_matrix: Tensor, y: Tensor, predictions: Tensor) -> Self {
        Self {
            hat_matrix,
            y,
            predictions,
        }
    }

    /// Compute leave-one-out cross-validation residuals.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.hat_matrix.device();
        let n: usize = self.y.shape().iter().product();

        if n == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(n)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
        }

        let params = Params { n: n as u32 };
        let params_buffer = device.create_uniform_buffer("LOO-CV Params", &params);

        ComputeDispatch::new(device, "LOO-CV")
            .shader(
                include_str!("../shaders/interpolation/loo_cv_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.hat_matrix.buffer())
            .storage_read(1, self.y.buffer())
            .storage_read(2, self.predictions.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(n as u32)
            .submit()?;

        Ok(Tensor::from_buffer(output_buffer, vec![n], device.clone()))
    }
}

impl Tensor {
    /// Compute LOO-CV residuals: (y - pred) / (1 - diag(H))
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn loo_cv(self, y: Self, predictions: Self) -> Result<Self> {
        LooCv::new(self, y, predictions).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_loo_cv() {
        let device = crate::device::test_pool::get_test_device().await;
        // Simple 2x2 case: H = [[0.5, 0.5], [0.5, 0.5]], y = [1, 2], pred = [0.8, 1.7]
        // LOO_0 = (1 - 0.8) / (1 - 0.5) = 0.2/0.5 = 0.4
        // LOO_1 = (2 - 1.7) / (1 - 0.5) = 0.3/0.5 = 0.6
        let hat_matrix = vec![0.5f32, 0.5, 0.5, 0.5];
        let y = vec![1.0f32, 2.0];
        let pred = vec![0.8f32, 1.7];

        let hat = Tensor::new(hat_matrix, vec![2, 2], device.clone());
        let y_t = Tensor::new(y, vec![2], device.clone());
        let pred_t = Tensor::new(pred, vec![2], device);

        let output = hat.loo_cv(y_t, pred_t).unwrap();
        let result = output.to_vec().unwrap();

        assert!((result[0] - 0.4).abs() < 1e-5);
        assert!((result[1] - 0.6).abs() < 1e-5);
    }
}
