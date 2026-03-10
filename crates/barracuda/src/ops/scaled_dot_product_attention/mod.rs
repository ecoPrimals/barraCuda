// SPDX-License-Identifier: AGPL-3.0-only
//! Scaled Dot-Product Attention - Transformer core operation
//!
//! ## Deep Debt Principles
//!
//! - **Pure GPU**: Multi-pass WGSL execution (no CPU fallbacks)
//! - **Zero hardcoding**: Runtime shape validation
//! - **Production-ready**: Complete implementation with proper error handling
//! - **Hardware-agnostic**: Pure WGSL for universal compute
//!
//! ## Evolution Path
//!
//! A single-kernel simplified variant is available via [`WGSL_SDPA_SINGLE_KERNEL`]
//! for prototyping. The production multi-pass implementation (3 passes) is preferred
//! for memory efficiency and numerical stability.
//!
//! ## Algorithm
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//! ```
//!
//! Where:
//! - Q: Query matrix [batch, heads, `seq_len`, `head_dim`]
//! - K: Key matrix [batch, heads, `seq_len`, `head_dim`]
//! - V: Value matrix [batch, heads, `seq_len`, `head_dim`]
//! - `d_k`: Dimension of keys (`head_dim`)
//!
//! ## Multi-Pass Execution
//!
//! 1. **Pass 1**: Compute Q @ K^T scores (`attention_matmul.wgsl`)
//! 2. **Pass 2**: Apply softmax to scores (`attention_softmax.wgsl`)
//! 3. **Pass 3**: Apply attention weights to values (`attention_apply.wgsl`)
//!
//! ## Reference
//!
//! "Attention is All You Need" (Vaswani et al., 2017)
//! <https://arxiv.org/abs/1706.03762>

mod compute;

#[cfg(test)]
mod tests;

use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static WGSL_SDPA_SINGLE_KERNEL_F64: &str =
    include_str!("../../shaders/attention/scaled_dot_product_attention_f64.wgsl");
static WGSL_SDPA_SINGLE_KERNEL_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_SDPA_SINGLE_KERNEL_F64.to_string());

/// Single-kernel SDPA shader for prototyping (simplified, non-production).
///
/// The production multi-pass implementation above is preferred; this constant
/// exposes the simplified variant for experimentation and testing.
#[must_use]
pub fn wgsl_sdpa_single_kernel() -> &'static str {
    &WGSL_SDPA_SINGLE_KERNEL_F32
}

/// Attention parameters for shader uniform binding.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AttentionParams {
    /// Batch dimension size.
    pub batch_size: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Query sequence length.
    pub q_seq_len: u32,
    /// Key/value sequence length.
    pub kv_seq_len: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Padding for alignment.
    pub _padding: [u32; 3],
}

/// Scaled dot-product attention operation
pub struct ScaledDotProductAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
}

impl ScaledDotProductAttention {
    /// Create a new scaled dot-product attention operation
    /// # Arguments
    /// - `query`: Query tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `key`: Key tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `value`: Value tensor [batch, heads, `seq_len`, `head_dim`]
    /// # Returns
    /// Result containing the operation struct, or error if shapes are invalid
    /// # Errors
    /// Returns [`Err`] if inputs are not 4D, shapes do not match, or tensors are on different devices.
    pub fn new(query: Tensor, key: Tensor, value: Tensor) -> Result<Self> {
        // Validate shapes match
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
        let num_heads = q_shape[1];
        let seq_len = q_shape[2];
        let head_dim = q_shape[3];

        // Validate all shapes match
        if k_shape != q_shape || v_shape != q_shape {
            return Err(crate::error::BarracudaError::shape_mismatch(
                q_shape.to_vec(),
                if k_shape == q_shape {
                    v_shape.to_vec()
                } else {
                    k_shape.to_vec()
                },
            ));
        }

        // Validate devices match (compare Arc pointers)
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
            num_heads,
            seq_len,
            head_dim,
        })
    }

    /// Get WGSL shader for attention matrix multiplication (Pass 1)
    pub(super) fn wgsl_shader_matmul() -> &'static str {
        &crate::ops::attention::ATTENTION_MATMUL_F32
    }

    /// Get WGSL shader for attention softmax (Pass 2)
    pub(super) fn wgsl_shader_softmax() -> &'static str {
        &crate::ops::attention::ATTENTION_SOFTMAX_F32
    }

    /// Get WGSL shader for attention apply (Pass 3)
    pub(super) fn wgsl_shader_apply() -> &'static str {
        &crate::ops::attention::ATTENTION_APPLY_F32
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

    /// Get number of heads
    pub(super) fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get sequence length
    pub(super) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get head dimension
    pub(super) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Execute the scaled dot-product attention operation
    /// Performs multi-pass execution:
    /// 1. Compute Q @ K^T scores
    /// 2. Apply softmax to scores
    /// 3. Apply attention weights to values
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        compute::execute_scaled_dot_product_attention(self)
    }
}

impl Tensor {
    /// Scaled dot-product attention
    /// Computes: Attention(Q, K, V) = softmax(QK^T / `sqrt(d_k)`) * V
    /// # Arguments
    /// - `key`: Key tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `value`: Value tensor [batch, heads, `seq_len`, `head_dim`]
    /// # Returns
    /// Output tensor [batch, heads, `seq_len`, `head_dim`]
    /// # Errors
    /// Returns [`Err`] if shape validation fails or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn scaled_dot_product_attention(self, key: Tensor, value: Tensor) -> Result<Self> {
        ScaledDotProductAttention::new(self, key, value)?.execute()
    }
}
