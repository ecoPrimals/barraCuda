// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-Head Attention - Complete attention layer with projections
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
//! ```text
//! MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
//! where head_i = Attention(Q*W^Q_i, K*W^K_i, V*W^V_i)
//! ```
//!
//! This is the complete attention mechanism used in transformers,
//! including all projection matrices.
//!
//! ## Multi-Pass Execution
//!
//! 1. **Pass 1-3**: Project Q, K, V through weight matrices (`mha_projection.wgsl`)
//! 2. **Pass 4**: Apply scaled dot-product attention
//! 3. **Pass 5**: Project concatenated heads through output matrix (`mha_output.wgsl`)

mod compute;

#[cfg(test)]
mod tests;

use crate::error::Result;
use crate::ops::scaled_dot_product_attention::ScaledDotProductAttention;
use crate::tensor::Tensor;

/// MHA projection parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MHAProjectionParams {
    pub batch_size: u32,
    pub seq_len: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub head_dim: u32,
}

/// Multi-head attention operation
pub struct MultiHeadAttention {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention operation
    ///
    /// # Arguments
    /// - `query`: Query tensor [batch, seq_len, d_model]
    /// - `key`: Key tensor [batch, seq_len, d_model]
    /// - `value`: Value tensor [batch, seq_len, d_model]
    /// - `w_q`: Query projection weights [d_model, d_model]
    /// - `w_k`: Key projection weights [d_model, d_model]
    /// - `w_v`: Value projection weights [d_model, d_model]
    /// - `w_o`: Output projection weights [d_model, d_model]
    /// - `num_heads`: Number of attention heads
    ///
    /// # Returns
    /// Result containing the operation struct, or error if shapes are invalid
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
        // Validate input shapes
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "All inputs must be 3D tensors [batch, seq_len, d_model]".to_string(),
            });
        }

        let batch_size = q_shape[0];
        let seq_len = q_shape[1];
        let d_model = q_shape[2];

        // Validate all input shapes match
        if k_shape != q_shape || v_shape != q_shape {
            return Err(crate::error::BarracudaError::shape_mismatch(
                q_shape.to_vec(),
                if k_shape != q_shape {
                    k_shape.to_vec()
                } else {
                    v_shape.to_vec()
                },
            ));
        }

        // Validate d_model is divisible by num_heads
        if !d_model.is_multiple_of(num_heads) {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
                ),
            });
        }

        let head_dim = d_model / num_heads;

        // Validate weight shapes
        let w_q_shape = w_q.shape();
        let w_k_shape = w_k.shape();
        let w_v_shape = w_v.shape();
        let w_o_shape = w_o.shape();

        let expected_weight_shape = vec![d_model, d_model];
        if w_q_shape != expected_weight_shape
            || w_k_shape != expected_weight_shape
            || w_v_shape != expected_weight_shape
            || w_o_shape != expected_weight_shape
        {
            return Err(crate::error::BarracudaError::shape_mismatch(
                expected_weight_shape,
                w_q_shape.to_vec(),
            ));
        }

        // Validate devices match
        use std::sync::Arc;
        if !Arc::ptr_eq(query.device(), key.device())
            || !Arc::ptr_eq(query.device(), value.device())
            || !Arc::ptr_eq(query.device(), w_q.device())
            || !Arc::ptr_eq(query.device(), w_k.device())
            || !Arc::ptr_eq(query.device(), w_v.device())
            || !Arc::ptr_eq(query.device(), w_o.device())
        {
            return Err(crate::error::BarracudaError::device(
                "All tensors must be on the same device",
            ));
        }

        Ok(Self {
            query,
            key,
            value,
            w_q,
            w_k,
            w_v,
            w_o,
            batch_size,
            seq_len,
            d_model,
            num_heads,
            head_dim,
        })
    }

    /// Execute the multi-head attention operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query.device();

        // Create command encoder for all passes
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MultiHeadAttention Encoder"),
            });

        // ═══════════════════════════════════════════════════════════════
        // PASS 1-3: Project Q, K, V through weight matrices
        // ═══════════════════════════════════════════════════════════════
        let q_proj_buffer =
            compute::execute_projection(&self, device, &self.query, &self.w_q, &mut encoder)?;
        let k_proj_buffer =
            compute::execute_projection(&self, device, &self.key, &self.w_k, &mut encoder)?;
        let v_proj_buffer =
            compute::execute_projection(&self, device, &self.value, &self.w_v, &mut encoder)?;

        // Submit projection passes
        device.submit_and_poll(Some(encoder.finish()));

        // ═══════════════════════════════════════════════════════════════
        // PASS 4: Apply scaled dot-product attention
        // ═══════════════════════════════════════════════════════════════
        // Create tensors from buffers for attention
        let q_proj = Tensor::from_buffer(
            q_proj_buffer,
            vec![self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            device.clone(),
        );
        let k_proj = Tensor::from_buffer(
            k_proj_buffer,
            vec![self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            device.clone(),
        );
        let v_proj = Tensor::from_buffer(
            v_proj_buffer,
            vec![self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            device.clone(),
        );

        // Apply attention
        let attention_output = ScaledDotProductAttention::new(q_proj, k_proj, v_proj)?.execute()?;

        // ═══════════════════════════════════════════════════════════════
        // PASS 5: Project concatenated heads through output matrix
        // ═══════════════════════════════════════════════════════════════
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MultiHeadAttention Output Encoder"),
            });

        let output_buffer = compute::execute_output_projection(
            &self,
            device,
            attention_output.buffer(),
            &mut encoder,
        )?;

        // Submit output projection pass
        device.submit_and_poll(Some(encoder.finish()));

        // Return output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.batch_size, self.seq_len, self.d_model],
            device.clone(),
        ))
    }

    /// Get batch size
    pub(super) fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get sequence length
    pub(super) fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Get d_model
    pub(super) fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get number of heads
    pub(super) fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get head dimension
    pub(super) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get output weight tensor
    pub(super) fn w_o(&self) -> &Tensor {
        &self.w_o
    }
}

// Note: Tensor::multi_head_attention is implemented in mha.rs
