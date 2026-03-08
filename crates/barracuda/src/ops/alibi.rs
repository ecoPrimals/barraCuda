// SPDX-License-Identifier: AGPL-3.0-or-later
//! `ALiBi` Position Encoding - GPU-accelerated
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (single-pass GPU)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for BLOOM, MPT)
//!
//! ## Algorithm
//!
//! ```text
//! ALiBi adds linear biases to attention scores:
//!
//! slope[h] = 2^(-(8*(h+1) / num_heads))
//! distance[i,j] = |i - j|
//! bias[h,i,j] = -slope[h] * distance[i,j]
//! output[b,h,i,j] = scores[b,h,i,j] + bias[h,i,j]
//! ```
//!
//! **Key Properties**:
//! - No learned parameters (like `RoPE`)
//! - Linear bias based on distance
//! - Head-specific slopes
//! - Enables "train short, test long" (extrapolates to longer sequences)
//!
//! **Used By**: BLOOM, MPT, `CodeGen`
//!
//! **Reference**: Press et al., 2021 - "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! // Compute attention scores first (before softmax)
//! let scores = query.matmul(&key.transpose())?;  // [batch, heads, seq, seq]
//!
//! // Apply ALiBi bias
//! let biased_scores = scores.alibi_position()?;
//!
//! // Then apply softmax and attend to values
//! let weights = biased_scores.softmax()?;
//! let output = weights.matmul(&value)?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/attention/alibi_position_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    SHADER_F64.to_string()
});

/// `ALiBi` parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AlibiParams {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    _padding: u32,
}

/// `ALiBi` Position Encoding operation
///
/// **Deep Debt**: Single-pass GPU, no learned parameters
pub struct AlibiPosition {
    scores: Tensor,
}

impl AlibiPosition {
    /// Create new `ALiBi` operation
    /// **Shape**: [batch, heads, `seq_len`, `seq_len`] (attention scores)
    /// # Errors
    /// Returns [`Err`] if scores is not 4D or the attention matrix is not square.
    pub fn new(scores: Tensor) -> Result<Self> {
        // Validate shape: must be 4D square attention matrix
        if scores.shape().len() != 4 {
            return Err(BarracudaError::shape_mismatch(
                scores.shape().to_vec(),
                vec![0, 0, 0, 0],
            ));
        }

        // Validate last two dims are equal (square attention matrix)
        let seq_len_1 = scores.shape()[2];
        let seq_len_2 = scores.shape()[3];
        if seq_len_1 != seq_len_2 {
            return Err(BarracudaError::invalid_op(
                "AlibiPosition",
                format!("Attention matrix must be square, got [{seq_len_1}, {seq_len_2}]"),
            ));
        }

        Ok(Self { scores })
    }

    /// WGSL shader source
    fn shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute `ALiBi` (single GPU pass)
    /// **Deep Debt**: Efficient single-pass, no intermediate buffers
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.scores.device();

        // Extract dimensions
        let shape = self.scores.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];

        // Create parameters
        let params = AlibiParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len: seq_len as u32,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("ALiBi Params", &params);

        let output_size = self.scores.len();
        let output_buffer = device.create_buffer_f32(output_size)?;

        let total = (batch_size * num_heads * seq_len * seq_len) as u32;

        ComputeDispatch::new(device, "ALiBi")
            .shader(Self::shader(), "main")
            .storage_read(0, self.scores.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(total)
            .submit()?;

        // Return output tensor (same shape as input)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, seq_len, seq_len],
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Apply `ALiBi` position encoding to attention scores
    /// **Deep Debt**: Essential for BLOOM, MPT, `CodeGen`
    /// # Arguments
    /// - Input: Attention scores [batch, heads, `seq_len`, `seq_len`]
    /// # Returns
    /// - Biased scores [batch, heads, `seq_len`, `seq_len`]
    /// # Example
    /// ```rust,ignore
    /// // Compute attention scores (before softmax)
    /// let scores = query.matmul(&key.transpose())?;
    /// // Apply ALiBi bias
    /// let biased = scores.alibi_position()?;  // BLOOM-style
    /// // Then softmax and apply to values
    /// let weights = biased.softmax()?;
    /// let output = weights.matmul(&value)?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if scores is not 4D, attention matrix is not square, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn alibi_position(self) -> Result<Self> {
        AlibiPosition::new(self)?.execute()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_alibi_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let heads = 2;
        let seq = 4;

        let scores = Tensor::from_vec_on(
            vec![1.0; batch * heads * seq * seq],
            vec![batch, heads, seq, seq],
            device,
        )
        .await
        .unwrap();

        let output = scores.alibi_position().unwrap();

        assert_eq!(output.shape(), &[batch, heads, seq, seq]);
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_alibi_single_token() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let heads = 1;
        let seq = 1;

        let scores = Tensor::from_vec_on(vec![5.0], vec![batch, heads, seq, seq], device)
            .await
            .unwrap();

        let output = scores.alibi_position().unwrap();

        // Distance=0, no bias added
        let data = output.to_vec().unwrap();
        assert!((data[0] - 5.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_alibi_bloom_dims() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // BLOOM-style dimensions
        let batch = 2;
        let heads = 8;
        let seq = 16;

        let scores = Tensor::from_vec_on(
            vec![0.0; batch * heads * seq * seq],
            vec![batch, heads, seq, seq],
            device,
        )
        .await
        .unwrap();

        let output = scores.alibi_position().unwrap();

        assert_eq!(output.shape(), &[batch, heads, seq, seq]);
        let data = output.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
        // Non-zero (bias should be applied)
        assert!(data.iter().any(|&x| x != 0.0));
    }

    #[tokio::test]
    async fn test_alibi_diagonal_zero() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch = 1;
        let heads = 1;
        let seq = 4;

        let scores = Tensor::from_vec_on(
            vec![1.0; batch * heads * seq * seq],
            vec![batch, heads, seq, seq],
            device,
        )
        .await
        .unwrap();

        let output = scores.alibi_position().unwrap();
        let data = output.to_vec().unwrap();

        // Diagonal elements (distance=0) should have no bias
        for i in 0..seq {
            let idx = i * seq + i;
            assert!((data[idx] - 1.0).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_alibi_shape_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Valid: square attention matrix
        let scores = Tensor::from_vec_on(vec![1.0; 2 * 4 * 4], vec![1, 2, 4, 4], device.clone())
            .await
            .unwrap();
        assert!(scores.alibi_position().is_ok());

        // Invalid: non-square matrix
        let scores = Tensor::from_vec_on(vec![1.0; 2 * 4 * 8], vec![1, 2, 4, 8], device)
            .await
            .unwrap();
        assert!(scores.alibi_position().is_err());
    }
}
