// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross Attention - Encoder-Decoder Attention
//!
//! **Deep Debt Principles**:
//! - ✅ Maximum code reuse (wrapper around validated attention!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for T5, BART)
//!
//! ## Algorithm
//!
//! ```text
//! CrossAttention(Q_decoder, K_encoder, V_encoder) = softmax(QK^T / sqrt(d_k)) * V
//! ```
//!
//! **Key Difference**: Q from decoder (`seq_len_q`), K/V from encoder (`seq_len_kv`)
//!
//! **Deep Debt Win**: Our attention already supports this! This is just a convenience API.
//!
//! **Used By**: T5, BART, Whisper, encoder-decoder transformers
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! // Decoder query: [batch, heads, dec_seq, dim]
//! let q_dec = Tensor::randn(vec![2, 8, 32, 64]).await?;
//!
//! // Encoder keys/values: [batch, heads, enc_seq, dim]
//! let k_enc = Tensor::randn(vec![2, 8, 128, 64]).await?;
//! let v_enc = Tensor::randn(vec![2, 8, 128, 64]).await?;
//!
//! // Cross attention (decoder attends to encoder)
//! let output = q_dec.cross_attention(&k_enc, &v_enc)?;
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

/// f64 canonical — cross attention matmul (sqrt).
static CROSS_ATTENTION_MATMUL_F64: &str =
    include_str!("../../shaders/math/cross_attention_matmul_f64.wgsl");
static CROSS_ATTENTION_MATMUL_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(CROSS_ATTENTION_MATMUL_F64)
});

const SHADER_F64: &str = include_str!("../../shaders/attention/cross_attention_apply_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

const SHADER_SOFTMAX_F64: &str =
    include_str!("../../shaders/activation/cross_attention_softmax_f64.wgsl");
static SHADER_SOFTMAX_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_SOFTMAX_F64)
});

#[cfg(test)]
mod tests;

/// Cross Attention parameters for WGSL shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CrossAttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub decoder_seq: u32,
    pub encoder_seq: u32,
    pub head_dim: u32,
    pub _padding: [u32; 3],
}

/// Cross Attention operation (encoder-decoder)
///
/// **Deep Debt**: Custom WGSL for asymmetric `seq_lens` (decoder != encoder)
pub struct CrossAttention {
    query: Tensor, // From decoder
    key: Tensor,   // From encoder
    value: Tensor, // From encoder
}

impl CrossAttention {
    /// Create new cross attention operation
    /// **Shapes**:
    /// - query (decoder): [batch, heads, `decoder_seq`, dim]
    /// - key (encoder): [batch, heads, `encoder_seq`, dim]
    /// - value (encoder): [batch, heads, `encoder_seq`, dim]
    /// # Errors
    /// Returns [`Err`] if tensors are not 4D, batch/heads/dim mismatch, or
    /// key and value have different sequence lengths.
    pub fn new(query: Tensor, key: Tensor, value: Tensor) -> Result<Self> {
        // Validate shapes: all must be 4D [batch, heads, seq, dim]
        if query.shape().len() != 4 || key.shape().len() != 4 || value.shape().len() != 4 {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                vec![0, 0, 0, 0],
            ));
        }

        // Validate: batch and heads must match across all tensors
        if query.shape()[0] != key.shape()[0] || query.shape()[0] != value.shape()[0] {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }

        if query.shape()[1] != key.shape()[1] || query.shape()[1] != value.shape()[1] {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }

        // Validate: head_dim must match
        if query.shape()[3] != key.shape()[3] || query.shape()[3] != value.shape()[3] {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }

        // Validate: K and V must have same seq_len (from encoder)
        if key.shape()[2] != value.shape()[2] {
            return Err(BarracudaError::shape_mismatch(
                key.shape().to_vec(),
                value.shape().to_vec(),
            ));
        }

        Ok(Self { query, key, value })
    }

    /// Shader references
    pub(super) fn shader_matmul() -> &'static str {
        &CROSS_ATTENTION_MATMUL_F32
    }

    pub(super) fn shader_softmax() -> &'static str {
        &SHADER_SOFTMAX_F32
    }

    pub(super) fn shader_apply() -> &'static str {
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
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Cross Attention (encoder-decoder attention)
    /// **Deep Debt**: Convenience wrapper for attention with asymmetric `seq_lens`
    /// # Arguments
    /// - `key`: Encoder keys [batch, heads, `encoder_seq`, dim]
    /// - `value`: Encoder values [batch, heads, `encoder_seq`, dim]
    /// # Returns
    /// - Output: [batch, heads, `decoder_seq`, dim]
    /// # Example
    /// ```rust,ignore
    /// // Decoder query
    /// let q = Tensor::randn(vec![2, 8, 32, 64]).await?;
    /// // Encoder keys/values
    /// let k = Tensor::randn(vec![2, 8, 128, 64]).await?;
    /// let v = Tensor::randn(vec![2, 8, 128, 64]).await?;
    /// // Cross attention (decoder attends to encoder)
    /// let output = q.cross_attention(&k, &v)?;  // T5/BART style
    /// ```
    /// # Errors
    /// Returns [`Err`] if shape validation fails, or buffer allocation/GPU
    /// dispatch fails (e.g. device lost or out of memory).
    pub fn cross_attention(self, key: &Self, value: &Self) -> Result<Self> {
        CrossAttention::new(self, key.clone(), value.clone())?.execute()
    }
}
