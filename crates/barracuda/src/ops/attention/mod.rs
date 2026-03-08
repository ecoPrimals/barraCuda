// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scaled Dot-Product Attention - GPU-accelerated implementation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL multi-pass implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//! ```
//!
//! **Implementation**: 3-pass GPU execution
//! 1. Pass 1: Compute QK^T scores (matrix multiplication)
//! 2. Pass 2: Apply softmax to scores (row-wise)
//! 3. Pass 3: Apply weights to values (weighted sum)
//!
//! **Reference**: "Attention is All You Need" (Vaswani et al., 2017)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let query = Tensor::randn(vec![2, 8, 128, 64]).await?;  // [batch, heads, seq, dim]
//! let key = Tensor::randn(vec![2, 8, 128, 64]).await?;
//! let value = Tensor::randn(vec![2, 8, 128, 64]).await?;
//!
//! let output = query.attention(&key, &value)?;
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

/// f64 canonical — attention matmul (sqrt).
pub(crate) static ATTENTION_MATMUL_F64: &str =
    include_str!("../../shaders/math/attention_matmul_f64.wgsl");
pub(crate) static ATTENTION_MATMUL_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| {
        ATTENTION_MATMUL_F64.to_string()
    });

/// f64 canonical — SDPA scores (sqrt).
pub(crate) const SDPA_SCORES_F64: &str =
    include_str!("../../shaders/attention/sdpa_scores_f64.wgsl");
pub(crate) static SDPA_SCORES_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    SDPA_SCORES_F64.to_string()
});

/// f64 canonical — f32 derived via `downcast_f64_to_f32` when needed.
pub(crate) const ATTENTION_APPLY_F64: &str =
    include_str!("../../shaders/attention/attention_apply_f64.wgsl");
pub(crate) static ATTENTION_APPLY_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| {
        ATTENTION_APPLY_F64.to_string()
    });

/// f64 canonical — attention softmax (exp).
pub(crate) const ATTENTION_SOFTMAX_F64: &str =
    include_str!("../../shaders/activation/attention_softmax_f64.wgsl");
pub(crate) static ATTENTION_SOFTMAX_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| {
        ATTENTION_SOFTMAX_F64.to_string()
    });

#[cfg(test)]
mod tests;

/// Attention parameters for WGSL shaders
///
/// Supports both self-attention (`q_seq_len` == `kv_seq_len`) and cross-attention
/// (`q_seq_len` != `kv_seq_len`). The score matrix is [B, H, `q_seq_len`, `kv_seq_len`].
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
    pub _padding: [u32; 3],
}

/// Scaled dot-product attention operation
///
/// **Multi-pass GPU implementation**:
/// - Pass 1: QK^T (attention scores)
/// - Pass 2: Softmax (attention weights)
/// - Pass 3: Apply to V (output)
pub struct Attention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
}

impl Attention {
    /// Create new attention operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes are not 4D or `batch/heads/head_dim` mismatch.
    pub fn new(query: Tensor, key: Tensor, value: Tensor) -> Result<Self> {
        // Validate shapes: all must be [batch, heads, seq_len, head_dim]
        let q_ndim = query.shape().len();
        let k_ndim = key.shape().len();
        let v_ndim = value.shape().len();

        if q_ndim != 4 {
            return Err(BarracudaError::invalid_op(
                "attention",
                format!(
                    "query requires 4D tensor [batch, heads, seq_len, head_dim], got {q_ndim}D"
                ),
            ));
        }
        if k_ndim != 4 {
            return Err(BarracudaError::invalid_op(
                "attention",
                format!("key requires 4D tensor [batch, heads, seq_len, head_dim], got {k_ndim}D"),
            ));
        }
        if v_ndim != 4 {
            return Err(BarracudaError::invalid_op(
                "attention",
                format!(
                    "value requires 4D tensor [batch, heads, seq_len, head_dim], got {v_ndim}D"
                ),
            ));
        }

        // Cross-attention: Q seq_len may differ from K/V seq_len, but
        // batch, heads, and head_dim must match. K and V must be identical shape.
        if query.shape()[0] != key.shape()[0]
            || query.shape()[1] != key.shape()[1]
            || query.shape()[3] != key.shape()[3]
        {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }
        if key.shape() != value.shape() {
            return Err(BarracudaError::shape_mismatch(
                key.shape().to_vec(),
                value.shape().to_vec(),
            ));
        }

        Ok(Self { query, key, value })
    }

    /// Pass 1 shader: Compute QK^T scores
    pub(crate) fn shader_matmul() -> &'static str {
        &ATTENTION_MATMUL_F32
    }

    /// Pass 2 shader: Apply softmax
    pub(crate) fn shader_softmax() -> &'static str {
        &ATTENTION_SOFTMAX_F32
    }

    /// Pass 3 shader: Apply weights to values
    pub(crate) fn shader_apply() -> &'static str {
        &ATTENTION_APPLY_F32
    }

    /// Get query tensor
    pub(crate) fn query(&self) -> &Tensor {
        &self.query
    }

    /// Get key tensor
    pub(crate) fn key(&self) -> &Tensor {
        &self.key
    }

    /// Get value tensor
    pub(crate) fn value(&self) -> &Tensor {
        &self.value
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Scaled dot-product attention
    ///
    /// # Arguments
    ///
    /// * `key` - Key tensor [batch, heads, `seq_len`, `head_dim`]
    /// * `value` - Value tensor [batch, heads, `seq_len`, `head_dim`]
    ///
    /// # Returns
    ///
    /// Output tensor [batch, heads, `seq_len`, `head_dim`]
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = query.attention(&key, &value)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn attention(self, key: &Self, value: &Self) -> Result<Self> {
        Attention::new(self, key.clone(), value.clone())?.execute()
    }
}
