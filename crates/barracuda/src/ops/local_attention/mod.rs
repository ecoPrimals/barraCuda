// SPDX-License-Identifier: AGPL-3.0-only
//! Local Attention - GPU-accelerated windowed attention
//!
//! **Deep Debt Principles**:
//! - ✅ Composition over duplication (reuses 2/3 attention shaders!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Memory-efficient (windowed attention for long sequences)
//!
//! ## Algorithm
//!
//! ```text
//! Local window: position i only attends to positions within window
//! window[i] = [max(0, i - half_window), min(seq_len, i + half_window + 1)]
//! This reduces computation: O(n²) → O(n*w) where w is window size
//! ```
//!
//! **Implementation**: 3-pass GPU execution (reuses 2 attention shaders!)
//! 1. Pass 1: Compute QK^T scores (reuse `attention_matmul.wgsl` ✅)
//! 2. Pass 2: Apply softmax with local window mask (NEW: `local_attention_softmax.wgsl`)
//! 3. Pass 3: Apply weights to values (reuse `attention_apply.wgsl` ✅)
//!
//! **Deep Debt**: Maximum code reuse - only 1 new shader for window masking!
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let q = Tensor::randn(vec![2, 8, 1024, 64]).await?;  // Long sequence!
//! let k = Tensor::randn(vec![2, 8, 1024, 64]).await?;
//! let v = Tensor::randn(vec![2, 8, 1024, 64]).await?;
//!
//! let output = q.local_attention(&k, &v, 4)?;  // window_size=4
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

#[cfg(test)]
mod tests;

const SHADER_LOCAL_SOFTMAX_F64: &str =
    include_str!("../../shaders/activation/local_attention_softmax_f64.wgsl");
static SHADER_LOCAL_SOFTMAX_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_LOCAL_SOFTMAX_F64.to_string());

/// Local attention parameters for WGSL shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LocalAttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub window_size: u32,
    pub _padding: [u32; 3],
}

/// Local attention operation
///
/// **Deep Debt**: Composes validated attention shaders + local window mask shader
pub struct LocalAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    window_size: usize,
}

impl LocalAttention {
    /// Create new local attention operation
    /// # Arguments
    /// - `window_size`: Size of attention window (must be > 0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    /// Also returns [`Err`] if shape validation fails.
    pub fn new(query: Tensor, key: Tensor, value: Tensor, window_size: usize) -> Result<Self> {
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

        if window_size == 0 {
            return Err(BarracudaError::invalid_op(
                "LocalAttention",
                "window_size must be > 0",
            ));
        }

        Ok(Self {
            query,
            key,
            value,
            window_size,
        })
    }

    /// Pass 1 shader: Compute QK^T scores (REUSED from attention ✅)
    pub(super) fn shader_matmul() -> &'static str {
        &crate::ops::attention::ATTENTION_MATMUL_F32
    }

    /// Pass 2 shader: Apply softmax with local window mask (NEW - only shader needed!)
    pub(super) fn shader_local_softmax() -> &'static str {
        &SHADER_LOCAL_SOFTMAX_F32
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

    /// Get window size
    pub(super) fn window_size(&self) -> usize {
        self.window_size
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Local attention (windowed attention for long sequences)
    /// **Deep Debt**: Reuses 2/3 attention shaders + local window mask shader
    /// # Arguments
    /// - `key`: Key tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `value`: Value tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `window_size`: Size of attention window (must be > 0)
    /// # Returns
    /// Output tensor [batch, heads, `seq_len`, `head_dim`]
    /// # Example
    /// ```rust,ignore
    /// let q = Tensor::randn(vec![2, 8, 1024, 64]).await?;  // Long sequence
    /// let k = Tensor::randn(vec![2, 8, 1024, 64]).await?;
    /// let v = Tensor::randn(vec![2, 8, 1024, 64]).await?;
    /// let output = q.local_attention(&k, &v, 4)?;  // window_size=4
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn local_attention(self, key: &Self, value: &Self, window_size: usize) -> Result<Self> {
        LocalAttention::new(self, key.clone(), value.clone(), window_size)?.execute()
    }
}
