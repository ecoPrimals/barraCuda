// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross Attention - Encoder-decoder attention
//!
//! ## Deep Debt Principles
//!
//! - **Production-critical**: Required for seq2seq, encoder-decoder transformers
//! - **Asymmetric**: Q from decoder, K/V from encoder
//! - **Complete**: Proper implementation with different sequence lengths
//!
//! ## Algorithm
//!
//! ```text
//! CrossAttention(Q_decoder, K_encoder, V_encoder) = softmax(QK^T / sqrt(d_k)) * V
//! ```
//!
//! Key difference: Q has different `seq_len` than K/V
//! Use: T5, BART, encoder-decoder transformers

use crate::error::{BarracudaError, Result};

/// Cross attention between encoder and decoder
///
/// ## Usage
///
/// ```no_run
/// use barracuda::ops::cross_attention::*;
///
/// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let batch = 2;
/// let heads = 8;
/// let decoder_len = 32;  // Decoder sequence
/// let encoder_len = 128; // Encoder sequence (can differ!)
/// let head_dim = 64;
///
/// let q_size = batch * heads * decoder_len * head_dim;
/// let kv_size = batch * heads * encoder_len * head_dim;
///
/// let query = vec![0.5; q_size];
/// let key = vec![0.5; kv_size];
/// let value = vec![1.0; kv_size];
///
/// let output = cross_attention(
///     device, queue,
///     &query, &key, &value,
///     batch, heads, decoder_len, encoder_len, head_dim
/// ).await.unwrap();
/// # }
/// ```
///
/// # Errors
///
/// Returns [`Err`] if query, key, or value lengths do not match expected
/// dimensions for the given batch, heads, and sequence lengths.
pub async fn cross_attention(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    query: &[f32], // [batch, heads, decoder_len, head_dim]
    key: &[f32],   // [batch, heads, encoder_len, head_dim]
    value: &[f32], // [batch, heads, encoder_len, head_dim]
    batch_size: usize,
    num_heads: usize,
    decoder_len: usize,
    encoder_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let q_size = batch_size * num_heads * decoder_len * head_dim;
    let kv_size = batch_size * num_heads * encoder_len * head_dim;

    if query.len() != q_size || key.len() != kv_size || value.len() != kv_size {
        return Err(BarracudaError::InvalidInput {
            message: "Dimension mismatch".to_string(),
        });
    }

    let mut output = vec![0.0f32; q_size];
    let scale = (head_dim as f32).sqrt();

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..decoder_len {
                // Compute attention scores with encoder
                let mut scores = vec![0.0f32; encoder_len];

                for (j, score_slot) in scores.iter_mut().enumerate() {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_idx = b * num_heads * decoder_len * head_dim
                            + h * decoder_len * head_dim
                            + i * head_dim
                            + d;
                        let k_idx = b * num_heads * encoder_len * head_dim
                            + h * encoder_len * head_dim
                            + j * head_dim
                            + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    *score_slot = score / scale;
                }

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Apply to encoder values
                for d in 0..head_dim {
                    let mut weighted_sum = 0.0;
                    for (j, &s_j) in scores.iter().enumerate() {
                        let v_idx = b * num_heads * encoder_len * head_dim
                            + h * encoder_len * head_dim
                            + j * head_dim
                            + d;
                        weighted_sum += s_j * value[v_idx];
                    }
                    let out_idx = b * num_heads * decoder_len * head_dim
                        + h * decoder_len * head_dim
                        + i * head_dim
                        + d;
                    output[out_idx] = weighted_sum;
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_cross_attention_basic() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let batch = 1;
        let heads = 2;
        let dec_len = 3;
        let enc_len = 5;
        let dim = 4;

        let q_size = batch * heads * dec_len * dim;
        let kv_size = batch * heads * enc_len * dim;

        let query = vec![0.5; q_size];
        let key = vec![0.5; kv_size];
        let value = vec![1.0; kv_size];

        let output = cross_attention(
            device, queue, &query, &key, &value, batch, heads, dec_len, enc_len, dim,
        )
        .await
        .unwrap();
        assert_eq!(output.len(), q_size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_cross_attention_edge_cases() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Single token decoder, single token encoder
        let batch = 1;
        let heads = 1;
        let dec_len = 1;
        let enc_len = 1;
        let dim = 4;

        let q_size = batch * heads * dec_len * dim;
        let kv_size = batch * heads * enc_len * dim;

        let query = vec![1.0; q_size];
        let key = vec![1.0; kv_size];
        let value = vec![2.0; kv_size];

        let output = cross_attention(
            device, queue, &query, &key, &value, batch, heads, dec_len, enc_len, dim,
        )
        .await
        .unwrap();

        assert_eq!(output.len(), q_size);
        // With single encoder token, should attend fully to it
        assert!(output.iter().all(|&x| (x - 2.0).abs() < 0.1));
    }

    #[tokio::test]
    async fn test_cross_attention_boundary() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Asymmetric lengths (decoder shorter than encoder)
        let batch = 2;
        let heads = 4;
        let dec_len = 8;
        let enc_len = 32; // Much longer encoder
        let dim = 8;

        let q_size = batch * heads * dec_len * dim;
        let kv_size = batch * heads * enc_len * dim;

        let query = vec![0.7; q_size];
        let key = vec![0.5; kv_size];
        let value = vec![1.0; kv_size];

        let output = cross_attention(
            device, queue, &query, &key, &value, batch, heads, dec_len, enc_len, dim,
        )
        .await
        .unwrap();

        assert_eq!(output.len(), q_size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_cross_attention_large_batch() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // T5/BART style dimensions
        let batch = 4;
        let heads = 8;
        let dec_len = 16;
        let enc_len = 64;
        let dim = 64;

        let q_size = batch * heads * dec_len * dim;
        let kv_size = batch * heads * enc_len * dim;

        let query: Vec<f32> = (0..q_size).map(|i| (i % 100) as f32 * 0.01).collect();
        let key: Vec<f32> = (0..kv_size).map(|i| (i % 100) as f32 * 0.01).collect();
        let value: Vec<f32> = (0..kv_size).map(|i| (i % 100) as f32 * 0.01).collect();

        let output = cross_attention(
            device, queue, &query, &key, &value, batch, heads, dec_len, enc_len, dim,
        )
        .await
        .unwrap();

        assert_eq!(output.len(), q_size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_cross_attention_precision() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Test attention distribution
        let batch = 1;
        let heads = 1;
        let dec_len = 2;
        let enc_len = 3;
        let dim = 2;

        let q_size = batch * heads * dec_len * dim;

        let query = vec![1.0, 0.0, 0.0, 1.0]; // 2 decoder tokens
        let key = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]; // 3 encoder tokens
        let value = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]; // Distinct values

        let output = cross_attention(
            device, queue, &query, &key, &value, batch, heads, dec_len, enc_len, dim,
        )
        .await
        .unwrap();

        assert_eq!(output.len(), q_size);
        assert!(output.iter().all(|&x| x.is_finite()));
        // Output should be a weighted combination of encoder values
        assert!(output.iter().all(|&x| (0.0..=3.0).contains(&x)));
    }
}
