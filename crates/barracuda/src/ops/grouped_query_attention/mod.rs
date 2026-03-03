// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grouped Query Attention - LLaMA-style efficient attention
//!
//! ## Deep Debt Principles
//!
//! - **Pure GPU**: Multi-pass WGSL execution (no CPU fallbacks)
//! - **Zero hardcoding**: Runtime shape validation
//! - **Production-ready**: Complete implementation with proper error handling
//! - **Hardware-agnostic**: Pure WGSL for universal compute
//!
//! ## Algorithm
//!
//! Multi-Query Attention variant where queries have multiple heads
//! but keys/values share heads across groups.
//!
//! Memory efficient: Reduces KV cache size for inference.
//! Reference: LLaMA, LLaMA-2 (Meta AI)
//!
//! ## Multi-Pass Execution
//!
//! Similar to scaled dot-product attention but with grouped key/value heads:
//! 1. Compute Q @ K^T scores (adapted for grouped queries)
//! 2. Apply softmax to scores
//! 3. Apply attention weights to values

use crate::error::Result;
use crate::tensor::Tensor;

mod compute;

const SHADER_F64: &str = include_str!("../../shaders/attention/gqa_apply_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

const SHADER_SOFTMAX_F64: &str = include_str!("../../shaders/activation/gqa_softmax_f64.wgsl");
static SHADER_SOFTMAX_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_SOFTMAX_F64)
});

#[cfg(test)]
mod tests;

/// GQA attention parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GQAParams {
    pub batch_size: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub heads_per_group: u32,
}

/// Grouped Query Attention operation
pub struct GroupedQueryAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    batch_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    heads_per_group: usize,
}

impl GroupedQueryAttention {
    /// Create a new grouped query attention operation
    ///
    /// # Arguments
    /// - `query`: Query tensor [batch, num_q_heads, seq_len, head_dim]
    /// - `key`: Key tensor [batch, num_kv_heads, seq_len, head_dim]
    /// - `value`: Value tensor [batch, num_kv_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Result containing the operation struct, or error if shapes are invalid
    pub fn new(query: Tensor, key: Tensor, value: Tensor) -> Result<Self> {
        // Validate shapes
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "All inputs must be 4D tensors [batch, heads, seq_len, head_dim]"
                    .to_string(),
            });
        }

        let batch_size = q_shape[0];
        let num_q_heads = q_shape[1];
        let seq_len = q_shape[2];
        let head_dim = q_shape[3];

        // Validate key/value shapes
        let num_kv_heads = k_shape[1];
        if k_shape[0] != batch_size || k_shape[2] != seq_len || k_shape[3] != head_dim {
            return Err(crate::error::BarracudaError::shape_mismatch(
                vec![batch_size, num_q_heads, seq_len, head_dim],
                k_shape.to_vec(),
            ));
        }

        if v_shape != k_shape {
            return Err(crate::error::BarracudaError::shape_mismatch(
                k_shape.to_vec(),
                v_shape.to_vec(),
            ));
        }

        // Validate num_q_heads is divisible by num_kv_heads
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            });
        }

        let heads_per_group = num_q_heads / num_kv_heads;

        // Validate devices match
        use std::sync::Arc;
        if !Arc::ptr_eq(query.device(), key.device())
            || !Arc::ptr_eq(query.device(), value.device())
        {
            return Err(crate::error::BarracudaError::device(
                "All tensors must be on the same device",
            ));
        }

        Ok(Self {
            query,
            key,
            value,
            batch_size,
            num_q_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            heads_per_group,
        })
    }

    /// Get WGSL shader for GQA attention matrix multiplication (Pass 1)
    pub(super) fn wgsl_shader_matmul() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../../shaders/math/gqa_matmul_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Get WGSL shader for GQA attention softmax (Pass 2)
    pub(super) fn wgsl_shader_softmax() -> &'static str {
        &SHADER_SOFTMAX_F32
    }

    /// Get WGSL shader for GQA attention apply (Pass 3)
    pub(super) fn wgsl_shader_apply() -> &'static str {
        &SHADER_F32
    }

    /// Get query tensor
    pub(super) fn query(&self) -> &Tensor {
        &self.query
    }

    /// Get key tensor
    pub(super) fn key(&self) -> &Tensor {
        &self.key
    }

    /// Get value tensor
    pub(super) fn value(&self) -> &Tensor {
        &self.value
    }

    /// Get batch size
    pub(super) fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get number of query heads
    pub(super) fn num_q_heads(&self) -> usize {
        self.num_q_heads
    }

    /// Get number of key/value heads
    pub(super) fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get sequence length
    pub(super) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get head dimension
    pub(super) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get heads per group
    pub(super) fn heads_per_group(&self) -> usize {
        self.heads_per_group
    }
}

impl Tensor {
    /// Grouped Query Attention
    ///
    /// Computes attention with grouped key/value heads (efficient for inference).
    ///
    /// # Arguments
    /// - `key`: Key tensor [batch, num_kv_heads, seq_len, head_dim]
    /// - `value`: Value tensor [batch, num_kv_heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch, num_q_heads, seq_len, head_dim]
    pub fn grouped_query_attention(self, key: Tensor, value: Tensor) -> Result<Self> {
        GroupedQueryAttention::new(self, key, value)?.execute()
    }
}
