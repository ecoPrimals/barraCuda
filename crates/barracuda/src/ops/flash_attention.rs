// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flash Attention - Memory-efficient attention mechanism
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (hardware-agnostic)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Memory-efficient O(N) instead of O(N²)
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Flash Attention reduces memory complexity from O(N²) to O(N) by:
//! 1. Computing attention in blocks (tiling)
//! 2. Fusing operations to minimize memory access
//! 3. Using numerically stable softmax
//!
//! Reference: "`FlashAttention`: Fast and Memory-Efficient Exact Attention"
//! by Dao et al. (2022) - 2-4x faster than standard attention!
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let query = Tensor::randn(vec![seq_len, head_dim]).await?;
//! let key = Tensor::randn(vec![seq_len, head_dim]).await?;
//! let value = Tensor::randn(vec![seq_len, head_dim]).await?;
//!
//! let output = query.flash_attention(&key, &value, num_heads)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashAttentionParams {
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    scale: f32,
}

/// Flash Attention operation
pub struct FlashAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: u32,
}

impl FlashAttention {
    /// Create Flash Attention operation
    /// # Errors
    /// Returns [`Err`] if shape validation fails (e.g. head dimension mismatch).
    pub fn new(query: Tensor, key: Tensor, value: Tensor, num_heads: u32) -> Result<Self> {
        // Verify shapes match
        if query.shape() != key.shape() || query.shape() != value.shape() {
            return Err(BarracudaError::invalid_op(
                "flash_attention",
                format!(
                    "Q, K, V must have same shape, got Q: {:?}, K: {:?}, V: {:?}",
                    query.shape(),
                    key.shape(),
                    value.shape()
                ),
            ));
        }

        // Verify 2D shape [seq_len, head_dim]
        if query.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "flash_attention",
                format!(
                    "Requires 2D tensors [seq_len, head_dim], got shape: {:?}",
                    query.shape()
                ),
            ));
        }

        Ok(Self {
            query,
            key,
            value,
            num_heads,
        })
    }

    /// Execute flash attention
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query.device();
        let shape = self.query.shape();
        let seq_len = shape[0];
        let head_dim = shape[1];
        let size = seq_len * head_dim;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Calculate scale factor (1 / sqrt(head_dim))
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create parameters
        let params = FlashAttentionParams {
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            num_heads: self.num_heads,
            scale,
        };

        // Create uniform buffer for parameters
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flash Attention Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Flash Attention")
            .shader(
                include_str!("../shaders/attention/flash_attention_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.query.buffer())
            .storage_read(1, self.key.buffer())
            .storage_read(2, self.value.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return result tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_flash_attention_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let seq_len = 4;
        let head_dim = 8;

        // Create simple Q, K, V
        let q_data = vec![1.0; seq_len * head_dim];
        let k_data = vec![1.0; seq_len * head_dim];
        let v_data = vec![2.0; seq_len * head_dim];

        let query = Tensor::from_vec_on(q_data, vec![seq_len, head_dim], device.clone())
            .await
            .unwrap();
        let key = Tensor::from_vec_on(k_data, vec![seq_len, head_dim], device.clone())
            .await
            .unwrap();
        let value = Tensor::from_vec_on(v_data, vec![seq_len, head_dim], device)
            .await
            .unwrap();

        let output = FlashAttention::new(query, key, value, 1)
            .unwrap()
            .execute()
            .unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), seq_len * head_dim);

        // Output should be weighted sum of values (all should be close to 2.0)
        for &val in &result {
            assert!((val - 2.0).abs() < 1.0, "Expected ~2.0, got {val}");
        }
    }

    #[tokio::test]
    async fn test_flash_attention_shape_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        let q = Tensor::from_vec_on(vec![1.0; 16], vec![4, 4], device.clone())
            .await
            .unwrap();
        let k = Tensor::from_vec_on(vec![1.0; 12], vec![3, 4], device.clone())
            .await
            .unwrap();
        let v = Tensor::from_vec_on(vec![1.0; 16], vec![4, 4], device)
            .await
            .unwrap();

        // Should fail: mismatched shapes
        let result = FlashAttention::new(q, k, v, 1);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_flash_attention_small_sequence() {
        let device = crate::device::test_pool::get_test_device().await;
        let seq_len = 2;
        let head_dim = 4;

        let query = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![seq_len, head_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let key = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![seq_len, head_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let value = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![seq_len, head_dim],
            device,
        )
        .await
        .unwrap();

        let output = FlashAttention::new(query, key, value, 1)
            .unwrap()
            .execute()
            .unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), seq_len * head_dim);

        // Each position should attend to all positions (uniform attention)
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
