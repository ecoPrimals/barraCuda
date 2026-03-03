//! Causal Attention - GPU-accelerated with causal masking
//!
//! **Deep Debt Principles**:
//! - ✅ Composition over duplication (reuses attention shaders)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for GPT-style models)
//!
//! ## Algorithm
//!
//! ```text
//! Causal mask: position i can only attend to positions 0..=i
//! mask[i,j] = -inf if j > i, else 0
//! attention = softmax((QK^T / sqrt(d_k)) + mask) * V
//! ```
//!
//! **Implementation**: 3-pass GPU execution (reuses 2 attention shaders!)
//! 1. Pass 1: Compute QK^T scores (reuse attention_matmul.wgsl ✅)
//! 2. Pass 2: Apply softmax with causal mask (NEW: causal_attention_softmax.wgsl)
//! 3. Pass 3: Apply weights to values (reuse attention_apply.wgsl ✅)
//!
//! **Deep Debt**: Maximum code reuse - only 1 new shader for masking!
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let q = Tensor::randn(vec![2, 8, 128, 64]).await?;  // [batch, heads, seq, dim]
//! let k = Tensor::randn(vec![2, 8, 128, 64]).await?;
//! let v = Tensor::randn(vec![2, 8, 128, 64]).await?;
//!
//! let output = q.causal_attention(&k, &v)?;  // GPT-style autoregressive attention
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

#[cfg(test)]
mod tests;

const SHADER_CAUSAL_SOFTMAX_F64: &str =
    include_str!("../../shaders/activation/causal_attention_softmax_f64.wgsl");
static SHADER_CAUSAL_SOFTMAX_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_CAUSAL_SOFTMAX_F64)
});

/// Attention parameters for WGSL shaders (same as regular attention)
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

/// Causal attention operation
///
/// **Deep Debt**: Composes validated attention shaders + causal mask shader
pub struct CausalAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
}

impl CausalAttention {
    /// Create new causal attention operation
    pub fn new(query: Tensor, key: Tensor, value: Tensor) -> Result<Self> {
        // Validate shapes: all must be [batch, heads, seq_len, head_dim]
        if query.shape().len() != 4 || key.shape().len() != 4 || value.shape().len() != 4 {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                vec![0, 0, 0, 0],
            ));
        }

        if query.shape() != key.shape() || query.shape() != value.shape() {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }

        Ok(Self { query, key, value })
    }

    /// Pass 1 shader: Compute QK^T scores (REUSED from attention ✅)
    pub(super) fn shader_matmul() -> &'static str {
        &crate::ops::attention::ATTENTION_MATMUL_F32
    }

    /// Pass 2 shader: Apply softmax with causal mask (NEW - only shader needed!)
    pub(super) fn shader_causal_softmax() -> &'static str {
        &SHADER_CAUSAL_SOFTMAX_F32
    }

    /// Pass 3 shader: Apply weights to values (REUSED from attention ✅)
    pub(super) fn shader_apply() -> &'static str {
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
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Causal attention (GPT-style autoregressive masking)
    ///
    /// **Deep Debt**: Reuses 2/3 attention shaders + causal mask shader
    ///
    /// # Arguments
    /// - `key`: Key tensor [batch, heads, seq_len, head_dim]
    /// - `value`: Value tensor [batch, heads, seq_len, head_dim]
    ///
    /// # Returns
    /// Output tensor [batch, heads, seq_len, head_dim]
    ///
    /// # Example
    /// ```rust,ignore
    /// let q = Tensor::randn(vec![2, 8, 128, 64]).await?;
    /// let k = Tensor::randn(vec![2, 8, 128, 64]).await?;
    /// let v = Tensor::randn(vec![2, 8, 128, 64]).await?;
    ///
    /// let output = q.causal_attention(&k, &v)?;  // GPT-style
    /// ```
    pub fn causal_attention(self, key: &Self, value: &Self) -> Result<Self> {
        CausalAttention::new(self, key.clone(), value.clone())?.execute()
    }
}
