// SPDX-License-Identifier: AGPL-3.0-or-later
//! Center Loss for metric learning
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Learns class centers and penalizes intra-class variance

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CenterLossParams {
    batch_size: u32,
    feature_dim: u32,
    num_classes: u32,
    _padding: u32,
}

/// Center loss for metric learning: penalizes intra-class variance.
pub struct CenterLoss {
    features: Tensor,
    centers: Tensor,
    labels: Tensor,
}

impl CenterLoss {
    /// Create `CenterLoss` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(features: Tensor, centers: Tensor, labels: Tensor) -> Result<Self> {
        // Validate shapes
        if features.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!(
                    "features must be 2D [batch, feature_dim], got shape {:?}",
                    features.shape()
                ),
            ));
        }

        if centers.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!(
                    "centers must be 2D [num_classes, feature_dim], got shape {:?}",
                    centers.shape()
                ),
            ));
        }

        if labels.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!("labels must be 1D [batch], got shape {:?}", labels.shape()),
            ));
        }

        Ok(Self {
            features,
            centers,
            labels,
        })
    }

    /// Execute `CenterLoss` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.features.device();
        let features_shape = self.features.shape();
        let batch_size = features_shape[0];
        let feature_dim = features_shape[1];

        let output_buffer = device.create_buffer_f32(batch_size)?;

        let params = CenterLossParams {
            batch_size: batch_size as u32,
            feature_dim: feature_dim as u32,
            num_classes: self.centers.shape()[0] as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CenterLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "CenterLoss")
            .shader(include_str!("../shaders/loss/center_loss_f64.wgsl"), "main")
            .storage_read(0, self.features.buffer())
            .storage_read(1, self.centers.buffer())
            .storage_read(2, self.labels.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_center_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 4;
        let feature_dim = 3;
        let num_classes = 2;

        let features = Tensor::from_vec_on(
            vec![1.0; batch_size * feature_dim],
            vec![batch_size, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let centers = Tensor::from_vec_on(
            vec![0.5; num_classes * feature_dim],
            vec![num_classes, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let labels =
            Tensor::from_vec_on(vec![0.0, 1.0, 0.0, 1.0], vec![batch_size], device.clone())
                .await
                .unwrap();

        let result = CenterLoss::new(features, centers, labels)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size]);
    }
}
