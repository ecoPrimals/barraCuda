// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse Attention - GPU-accelerated with strided sparse pattern
//!
//! **Deep Debt Principles**:
//! - ✅ Composition over duplication (reuses 2/3 attention shaders!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Memory-efficient (sparse pattern for long sequences)
//!
//! ## Algorithm
//!
//! ```text
//! Sparse mask: position i only attends to positions j where j % stride == 0
//! This reduces computation: O(n²) → O(n²/stride)
//! Example stride=4: attend to [0, 4, 8, 12, 16, ...]
//! ```
//!
//! **Implementation**: 3-pass GPU execution (reuses 2 attention shaders!)
//! 1. Pass 1: Compute QK^T scores (reuse `attention_matmul.wgsl` ✅)
//! 2. Pass 2: Apply softmax with sparse mask (NEW: `sparse_attention_softmax.wgsl`)
//! 3. Pass 3: Apply weights to values (reuse `attention_apply.wgsl` ✅)
//!
//! **Deep Debt**: Maximum code reuse - only 1 new shader for sparse pattern!
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
//! let output = q.sparse_attention(&k, &v, 4)?;  // stride=4, attend to every 4th token
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

#[cfg(test)]
mod tests;

const SHADER_SPARSE_SOFTMAX_F64: &str =
    include_str!("../../shaders/activation/sparse_attention_softmax_f64.wgsl");
static SHADER_SPARSE_SOFTMAX_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_SPARSE_SOFTMAX_F64)
});

/// Sparse attention parameters for WGSL shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SparseAttentionParams {
    pub batch_size: u32,
    pub num_heads: u32,
    pub seq_len: u32,
    pub head_dim: u32,
    pub stride: u32,
    pub _padding: [u32; 3],
}

/// Sparse attention operation
///
/// **Deep Debt**: Composes validated attention shaders + sparse mask shader
pub struct SparseAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    stride: usize,
}

impl SparseAttention {
    /// Create new sparse attention operation
    /// # Arguments
    /// - `stride`: Attend to every stride-th position (stride=1 is full attention)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(query: Tensor, key: Tensor, value: Tensor, stride: usize) -> Result<Self> {
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

        if stride == 0 {
            return Err(BarracudaError::invalid_op(
                "SparseAttention",
                "stride must be > 0",
            ));
        }

        Ok(Self {
            query,
            key,
            value,
            stride,
        })
    }

    /// Pass 1 shader: Compute QK^T scores (REUSED from attention ✅)
    pub(super) fn shader_matmul() -> &'static str {
        &crate::ops::attention::ATTENTION_MATMUL_F32
    }

    /// Pass 2 shader: Apply softmax with sparse mask (NEW - only shader needed!)
    pub(super) fn shader_sparse_softmax() -> &'static str {
        &SHADER_SPARSE_SOFTMAX_F32
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

    /// Get stride
    pub(super) fn stride(&self) -> usize {
        self.stride
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Sparse attention (strided pattern for long sequences)
    /// **Deep Debt**: Reuses 2/3 attention shaders + sparse mask shader
    /// # Arguments
    /// - `key`: Key tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `value`: Value tensor [batch, heads, `seq_len`, `head_dim`]
    /// - `stride`: Attend to every stride-th position (1 = full attention)
    /// # Returns
    /// Output tensor [batch, heads, `seq_len`, `head_dim`]
    /// # Example
    /// ```rust,ignore
    /// let q = Tensor::randn(vec![2, 8, 1024, 64]).await?;  // Long sequence
    /// let k = Tensor::randn(vec![2, 8, 1024, 64]).await?;
    /// let v = Tensor::randn(vec![2, 8, 1024, 64]).await?;
    /// let output = q.sparse_attention(&k, &v, 4)?;  // stride=4
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn sparse_attention(self, key: &Self, value: &Self, stride: usize) -> Result<Self> {
        SparseAttention::new(self, key.clone(), value.clone(), stride)?.execute()
    }
}
