// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cosine embedding loss operation
//!
//! Measures similarity between embeddings using cosine similarity
//! Used in metric learning, face recognition, and contrastive learning

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CosineEmbeddingLossParams {
    size: u32,
    margin: f32,
    _padding: [u32; 2],
}

/// Cosine embedding loss operation
pub struct CosineEmbeddingLoss {
    input1: Tensor,
    input2: Tensor,
    label: Tensor,
    margin: f32,
}

impl CosineEmbeddingLoss {
    /// Create cosine embedding loss operation
    /// # Errors
    /// Returns [`Err`] if input1 and input2 shapes differ, or label is not scalar [1].
    pub fn new(input1: Tensor, input2: Tensor, label: Tensor, margin: f32) -> Result<Self> {
        if input1.shape() != input2.shape() {
            return Err(BarracudaError::invalid_op(
                "cosine_embedding_loss",
                format!(
                    "input1 shape {:?} must match input2 shape {:?}",
                    input1.shape(),
                    input2.shape()
                ),
            ));
        }

        if label.shape() != [1] {
            return Err(BarracudaError::invalid_op(
                "cosine_embedding_loss",
                format!("label must be scalar [1], got shape {:?}", label.shape()),
            ));
        }

        Ok(Self {
            input1,
            input2,
            label,
            margin,
        })
    }

    /// Execute cosine embedding loss on tensors
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input1.device();
        let size = self.input1.len();

        let output_buffer = device.create_buffer_f32(1)?;

        let params = CosineEmbeddingLossParams {
            size: size as u32,
            margin: self.margin,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CosineEmbeddingLoss Params"),
            size: std::mem::size_of::<CosineEmbeddingLossParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(device, "CosineEmbeddingLoss")
            .shader(
                include_str!("../shaders/loss/cosine_embedding_loss_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input1.buffer())
            .storage_read(1, self.input2.buffer())
            .storage_read(2, self.label.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cosine_embedding_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input1 = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let input2 = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        let label = Tensor::from_vec_on(vec![1.0], vec![1], device)
            .await
            .unwrap();

        let output = CosineEmbeddingLoss::new(input1, input2, label, 0.5)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0] >= 0.0);
    }
}
