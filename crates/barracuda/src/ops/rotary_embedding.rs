// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rotary Position Embedding (`RoPE`) - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Applies rotation to query/key pairs based on position.
//! Encodes relative position information without absolute position embeddings.
//!
//! Reference: `RoFormer` (Su et al., 2021), used in GPT-Neo, `LLaMA`, `PaLM`

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/attention/rotary_embedding_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Rotary Position Embedding operation
pub struct RotaryEmbedding {
    input: Tensor,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding operation
    /// **Shape**: [batch, `seq_len`, `num_heads`, `head_dim`]
    /// **Requirement**: `head_dim` must be even
    /// # Errors
    /// Returns [`Err`] if input is not 4D or if `head_dim` is not even.
    pub fn new(input: Tensor) -> Result<Self> {
        // Validate shape: must be 4D
        if input.shape().len() != 4 {
            return Err(BarracudaError::shape_mismatch(
                input.shape().to_vec(),
                vec![0, 0, 0, 0],
            ));
        }

        // Validate head_dim is even (required for pairwise rotation)
        let head_dim = input.shape()[3];
        if !head_dim.is_multiple_of(2) {
            return Err(BarracudaError::invalid_op(
                "RotaryEmbedding",
                format!("head_dim ({head_dim}) must be even for pairwise rotation"),
            ));
        }

        Ok(Self { input })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the rotary embedding operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let num_heads = shape[2];
        let head_dim = shape[3];
        let half_dim = head_dim / 2;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_size = self.input.len();
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            seq_len: u32,
            num_heads: u32,
            head_dim: u32,
            half_dim: u32,
            _padding: [u32; 3],
        }

        let params = Params {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            half_dim: half_dim as u32,
            _padding: [0; 3],
        };

        let params_buffer = device.create_uniform_buffer("RotaryEmbedding Params", &params);

        let total = batch_size * seq_len * num_heads * half_dim;

        ComputeDispatch::new(device, "RotaryEmbedding")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

// Note: Tensor::rotary_embedding() is implemented in rope.rs to avoid duplication

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rotary_embedding_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 4 * 2 * 8], vec![1, 4, 2, 8], device)
            .await
            .unwrap();

        let output = input.rotary_embedding().unwrap();
        assert_eq!(output.shape(), &[1, 4, 2, 8]);
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_rotary_embedding_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single position
        let input = Tensor::from_vec_on(vec![1.0; 2 * 8], vec![1, 1, 2, 8], device.clone())
            .await
            .unwrap();
        let output = input.rotary_embedding().unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 8]);

        // Single head
        let input = Tensor::from_vec_on(vec![1.0; 4 * 8], vec![1, 4, 1, 8], device.clone())
            .await
            .unwrap();
        let output = input.rotary_embedding().unwrap();
        assert_eq!(output.shape(), &[1, 4, 1, 8]);

        // Small head dimension
        let input = Tensor::from_vec_on(vec![1.0; 2 * 2 * 4], vec![1, 2, 2, 4], device)
            .await
            .unwrap();
        let output = input.rotary_embedding().unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 4]);
    }

    #[tokio::test]
    async fn test_rotary_embedding_shape_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Valid: even head_dim
        let input = Tensor::from_vec_on(vec![1.0; 4 * 2 * 8], vec![1, 4, 2, 8], device.clone())
            .await
            .unwrap();
        assert!(input.rotary_embedding().is_ok());

        // Invalid: odd head_dim
        let input = Tensor::from_vec_on(vec![1.0; 4 * 2 * 7], vec![1, 4, 2, 7], device)
            .await
            .unwrap();
        assert!(input.rotary_embedding().is_err());
    }
}
