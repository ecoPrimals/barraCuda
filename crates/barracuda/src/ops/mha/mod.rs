// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-Head Attention - GPU-accelerated implementation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL + composition (custom projection + validated attention)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! ```text
//! MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
//! where head_i = Attention(Q*W^Q_i, K*W^K_i, V*W^V_i)
//! ```
//!
//! **Implementation**: 5-pass GPU execution
//! 1. Pass 1: Project Q through W_q with head split: [B,S,D] → [B,H,S,D/H]
//! 2. Pass 2: Project K through W_k with head split: [B,S,D] → [B,H,S,D/H]
//! 3. Pass 3: Project V through W_v with head split: [B,S,D] → [B,H,S,D/H]
//! 4. Pass 4: Apply validated attention: [B,H,S,D/H] → [B,H,S,D/H]
//! 5. Pass 5: Concat heads + project through W_o: [B,H,S,D/H] → [B,S,D]
//!
//! **Deep Debt**: Custom WGSL for projections + reuse validated attention
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let q = Tensor::randn(vec![2, 128, 512]).await?;  // [batch, seq, d_model]
//! let k = Tensor::randn(vec![2, 128, 512]).await?;
//! let v = Tensor::randn(vec![2, 128, 512]).await?;
//!
//! // Projection weights [d_model, d_model]
//! let w_q = Tensor::randn(vec![512, 512]).await?;
//! let w_k = Tensor::randn(vec![512, 512]).await?;
//! let w_v = Tensor::randn(vec![512, 512]).await?;
//! let w_o = Tensor::randn(vec![512, 512]).await?;
//!
//! let output = q.multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, 8)?;
//! // output.shape() == [2, 128, 512]
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod projections;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;

/// MHA parameters for WGSL shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MhaParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub _padding: [u32; 3],
}

/// Multi-head attention operation
///
/// **Deep Debt**: Custom WGSL for projections + validated attention core
pub struct MultiHeadAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    num_heads: usize,
}

impl MultiHeadAttention {
    /// Create new multi-head attention operation
    ///
    /// **Shapes**:
    /// - query, key, value: [batch, seq_len, d_model]
    /// - w_q, w_k, w_v, w_o: [d_model, d_model]
    pub fn new(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        w_o: Tensor,
        num_heads: usize,
    ) -> Result<Self> {
        // Validate input shapes (must be 3D: [batch, seq, d_model])
        if query.shape().len() != 3 || key.shape().len() != 3 || value.shape().len() != 3 {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                vec![0, 0, 0], // Expected 3D
            ));
        }

        // For cross-attention: Q seq_len can differ from K/V seq_len
        // But batch and d_model must match, and K/V must match each other
        if query.shape()[0] != key.shape()[0] || query.shape()[0] != value.shape()[0] {
            return Err(BarracudaError::shape_mismatch(
                query.shape().to_vec(),
                key.shape().to_vec(),
            ));
        }

        if query.shape()[2] != key.shape()[2] || key.shape()[2] != value.shape()[2] {
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

        // Validate projection weights (must be 2D: [d_model, d_model])
        if w_q.shape().len() != 2
            || w_k.shape().len() != 2
            || w_v.shape().len() != 2
            || w_o.shape().len() != 2
        {
            return Err(BarracudaError::shape_mismatch(
                w_q.shape().to_vec(),
                vec![0, 0], // Expected 2D
            ));
        }

        let d_model = query.shape()[2];
        if w_q.shape() != [d_model, d_model]
            || w_k.shape() != [d_model, d_model]
            || w_v.shape() != [d_model, d_model]
            || w_o.shape() != [d_model, d_model]
        {
            return Err(BarracudaError::shape_mismatch(
                w_q.shape().to_vec(),
                vec![d_model, d_model],
            ));
        }

        // Validate num_heads divides d_model evenly
        if !d_model.is_multiple_of(num_heads) {
            return Err(BarracudaError::InvalidOperation {
                op: "multi_head_attention".to_string(),
                reason: format!("d_model ({d_model}) must be divisible by num_heads ({num_heads})"),
            });
        }

        Ok(Self {
            query,
            key,
            value,
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
        })
    }

    /// Execute multi-head attention on GPU
    pub fn execute(self) -> Result<Tensor> {
        let batch_size = self.query.shape()[0];
        let q_seq_len = self.query.shape()[1];
        let d_model = self.query.shape()[2];
        let head_dim = d_model / self.num_heads;

        let params = MhaParams {
            batch_size: batch_size as u32,
            seq_len: q_seq_len as u32,
            d_model: d_model as u32,
            num_heads: self.num_heads as u32,
            head_dim: head_dim as u32,
            _padding: [0; 3],
        };

        // ═══════════════════════════════════════════════════════════
        // PASS 1-3: Project Q, K, V through weights with head split
        // [B, S, D] + [D, D] → [B, H, S, D/H]
        // ═══════════════════════════════════════════════════════════

        let q_proj = self.project_with_head_split(&self.query, &self.w_q, &params)?;
        let k_proj = self.project_with_head_split(&self.key, &self.w_k, &params)?;
        let v_proj = self.project_with_head_split(&self.value, &self.w_v, &params)?;

        // ═══════════════════════════════════════════════════════════
        // PASS 4: Apply validated attention
        // [B, H, S, D/H] → [B, H, S, D/H]
        // ═══════════════════════════════════════════════════════════

        let attention_output = q_proj.attention(&k_proj, &v_proj)?;

        // ═══════════════════════════════════════════════════════════
        // PASS 5: Concat heads + project through output weight
        // [B, H, S, D/H] → [B, S, D]
        // ═══════════════════════════════════════════════════════════

        self.concat_and_project(&attention_output, &self.w_o, &params)
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR TRAIT IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Multi-head attention with learned projections
    ///
    /// **Deep Debt**: Composes validated operations (proven to work!)
    ///
    /// # Arguments
    /// - `key`: Key tensor [batch, seq_len, d_model]
    /// - `value`: Value tensor [batch, seq_len, d_model]
    /// - `w_q`, `w_k`, `w_v`, `w_o`: Projection weights [d_model, d_model]
    /// - `num_heads`: Number of attention heads
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, d_model]
    ///
    /// # Example
    /// ```rust,ignore
    /// let q = Tensor::randn(vec![2, 128, 512]).await?;
    /// let k = Tensor::randn(vec![2, 128, 512]).await?;
    /// let v = Tensor::randn(vec![2, 128, 512]).await?;
    ///
    /// let w_q = Tensor::randn(vec![512, 512]).await?;
    /// let w_k = Tensor::randn(vec![512, 512]).await?;
    /// let w_v = Tensor::randn(vec![512, 512]).await?;
    /// let w_o = Tensor::randn(vec![512, 512]).await?;
    ///
    /// let output = q.multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, 8)?;
    /// ```
    pub fn multi_head_attention(
        self,
        key: &Self,
        value: &Self,
        w_q: &Self,
        w_k: &Self,
        w_v: &Self,
        w_o: &Self,
        num_heads: usize,
    ) -> Result<Self> {
        MultiHeadAttention::new(
            self,
            key.clone(),
            value.clone(),
            w_q.clone(),
            w_k.clone(),
            w_v.clone(),
            w_o.clone(),
            num_heads,
        )?
        .execute()
    }
}
