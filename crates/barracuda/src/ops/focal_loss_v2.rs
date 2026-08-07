// SPDX-License-Identifier: AGPL-3.0-or-later
//! Focal Loss v2 - Enhanced focal loss with alpha balancing
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! Improved version with class balancing parameter.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Focal Loss v2 operation
pub struct FocalLossV2 {
    predictions: Tensor,
    targets: Tensor,
    alpha: f32,
    gamma: f32,
}

impl FocalLossV2 {
    /// Create a new focal loss v2 operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch or parameters are invalid (alpha not in [0, 1], gamma < 0).
    pub fn new(predictions: Tensor, targets: Tensor, alpha: f32, gamma: f32) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        // Validate parameters
        if !(0.0..=1.0).contains(&alpha) {
            return Err(BarracudaError::invalid_op(
                "FocalLossV2",
                format!("alpha must be in [0, 1], got {alpha}"),
            ));
        }

        if gamma < 0.0 {
            return Err(BarracudaError::invalid_op(
                "FocalLossV2",
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

    /// Execute the focal loss v2 operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            alpha: f32,
            gamma: f32,
            epsilon: f32,
            size: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            alpha: self.alpha,
            gamma: self.gamma,
            epsilon: 1e-7,
            size: size as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FocalLossV2 Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "FocalLossV2")
            .shader(
                include_str!("../shaders/loss/focal_loss_v2_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Output shape: same as input (element-wise loss)
        Ok(Tensor::from_buffer(
            output_buffer,
            self.predictions.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_focal_loss_v2_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let preds = Tensor::from_vec_on(vec![0.9, 0.1, 0.8], vec![3], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 0.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();
        let loss = FocalLossV2::new(preds, targets, 0.25, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = loss.to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[tokio::test]
    async fn test_focal_loss_v2_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Perfect predictions
        let preds = Tensor::from_vec_on(vec![1.0, 0.0, 1.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 0.0, 1.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let loss = FocalLossV2::new(preds, targets, 0.25, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = loss.to_vec().unwrap();
        assert!(result.iter().all(|&x| x < 0.1));
    }
}
