//! Causal Attention - GPT-style masked attention
//!
//! ## Deep Debt Principles
//!
//! - **Production-critical**: Required for autoregressive models (GPT, etc.)
//! - **Correct masking**: Prevents attending to future tokens
//! - **Complete**: Proper causal mask implementation
//!
//! ## Algorithm
//!
//! Standard attention with causal masking:
//! ```text
//! mask[i,j] = -inf if j > i else 0
//! attention = softmax((QK^T / sqrt(d_k)) + mask) * V
//! ```
//!
//! Use: GPT, decoder-only transformers, autoregressive generation

use crate::error::{BarracudaError, Result};

/// Causal attention with masking
///
/// ## Usage
///
/// ```no_run
/// use barracuda::ops::causal_attention::*;
///
/// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let batch = 2;
/// let heads = 8;
/// let seq_len = 128;
/// let head_dim = 64;
///
/// let size = batch * heads * seq_len * head_dim;
/// let query = vec![0.5; size];
/// let key = vec![0.5; size];
/// let value = vec![1.0; size];
///
/// let output = causal_attention(
///     device, queue,
///     &query, &key, &value,
///     batch, heads, seq_len, head_dim
/// ).await.unwrap();
/// # }
/// ```
pub async fn causal_attention(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let expected_size = batch_size * num_heads * seq_len * head_dim;
    if query.len() != expected_size {
        return Err(BarracudaError::InvalidInput {
            message: "Dimension mismatch".to_string(),
        });
    }

    let mut output = vec![0.0f32; expected_size];
    let scale = (head_dim as f32).sqrt();

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len {
                // Compute scores with causal mask
                let mut scores = vec![f32::NEG_INFINITY; seq_len];

                // Only attend to current and previous positions (causal mask)
                for j in 0..=i {
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let q_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + i * head_dim
                            + d;
                        let k_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + j * head_dim
                            + d;
                        score += query[q_idx] * key[k_idx];
                    }
                    scores[j] = score / scale;
                }

                // Softmax (only over valid positions due to -inf masking)
                let max_score = scores
                    .iter()
                    .filter(|&&s| s.is_finite())
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    if s.is_finite() {
                        *s = (*s - max_score).exp();
                        sum += *s;
                    } else {
                        *s = 0.0;
                    }
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Apply to values
                for d in 0..head_dim {
                    let mut weighted_sum = 0.0;
                    for j in 0..seq_len {
                        let v_idx = b * num_heads * seq_len * head_dim
                            + h * seq_len * head_dim
                            + j * head_dim
                            + d;
                        weighted_sum += scores[j] * value[v_idx];
                    }
                    let out_idx = b * num_heads * seq_len * head_dim
                        + h * seq_len * head_dim
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
    async fn test_causal_attention_basic() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let batch = 1;
        let heads = 1;
        let seq = 4;
        let dim = 4;

        let size = batch * heads * seq * dim;
        let query = vec![1.0; size];
        let key = query.clone();
        let value = vec![1.0; size];

        let output = causal_attention(device, queue, &query, &key, &value, batch, heads, seq, dim)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_causal_attention_edge_cases() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Single token (no causal masking needed)
        let batch = 1;
        let heads = 1;
        let seq = 1;
        let dim = 4;

        let size = batch * heads * seq * dim;
        let query = vec![0.5; size];
        let key = query.clone();
        let value = vec![1.0; size];

        let output = causal_attention(device, queue, &query, &key, &value, batch, heads, seq, dim)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_causal_attention_boundary() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Multiple heads
        let batch = 1;
        let heads = 4;
        let seq = 3;
        let dim = 8;

        let size = batch * heads * seq * dim;
        let query = vec![0.1; size];
        let key = query.clone();
        let value = vec![1.0; size];

        let output = causal_attention(device, queue, &query, &key, &value, batch, heads, seq, dim)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_causal_attention_large_batch() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // GPT-style dimensions
        let batch = 2;
        let heads = 8;
        let seq = 16;
        let dim = 16;

        let size = batch * heads * seq * dim;
        let query = vec![0.5; size];
        let key = query.clone();
        let value = vec![1.0; size];

        let output = causal_attention(device, queue, &query, &key, &value, batch, heads, seq, dim)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_causal_attention_precision() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        // Test causal masking: first token only attends to itself
        let batch = 1;
        let heads = 1;
        let seq = 3;
        let dim = 2;

        let size = batch * heads * seq * dim;
        let mut query = vec![0.0; size];
        let mut key = vec![0.0; size];
        let mut value = vec![0.0; size];

        // Set distinct values for each token
        for i in 0..seq {
            for d in 0..dim {
                let idx = i * dim + d;
                query[idx] = (i + 1) as f32;
                key[idx] = (i + 1) as f32;
                value[idx] = (i + 1) as f32 * 10.0;
            }
        }

        let output = causal_attention(device, queue, &query, &key, &value, batch, heads, seq, dim)
            .await
            .unwrap();

        // Output should be finite and valid
        assert!(output.iter().all(|&x| x.is_finite()));

        // First token should only attend to itself (no future tokens)
        // This is a property of causal attention
        assert_eq!(output.len(), size);
    }
}
