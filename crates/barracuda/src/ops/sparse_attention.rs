// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse Attention - Strided attention pattern
//!
//! Only attends to every k-th token (stride).
//! Reduces complexity for long sequences.

use crate::error::Result;

/// Strided sparse attention: attends only to every k-th token to reduce complexity.
///
/// # Arguments
/// * `query`, `key`, `value` - Attention tensors (batch, heads, seq_len, head_dim)
/// * `stride` - Attend to positions 0, stride, 2*stride, ...
pub async fn sparse_attention(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    stride: usize,
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
    let scale = (head_dim as f32).sqrt();

    for b in 0..batch_size {
        for h in 0..num_heads {
            for i in 0..seq_len {
                let mut scores = vec![f32::NEG_INFINITY; seq_len];

                // Attend to strided positions: 0, stride, 2*stride, ...
                for j in (0..seq_len).step_by(stride) {
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
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_sparse_attention_basic() {
        let Some(dev) = get_test_device().await else {
            return;
        };
        let size = 2 * 8 * 4;
        let q = vec![0.5; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 1, 2, 8, 4, 2)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_sparse_attention_edge_cases() {
        let Some(dev) = get_test_device().await else {
            return;
        };
        // Single head
        let size = 4 * 4;
        let q = vec![1.0; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 1, 1, 4, 4, 2)
            .await
            .unwrap();
        assert_eq!(output.len(), size);

        // Small sequence (stride 1 = full attention)
        let size = 2 * 4 * 4;
        let q = vec![0.5; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 1, 2, 4, 4, 1)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
    }

    #[tokio::test]
    async fn test_sparse_attention_boundary() {
        let Some(dev) = get_test_device().await else {
            return;
        };
        // Large stride
        let size = 2 * 16 * 8;
        let q = vec![0.5; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 1, 2, 16, 8, 4)
            .await
            .unwrap();
        assert_eq!(output.len(), size);

        // Stride equals sequence length (attend to first token only)
        let size = 8 * 4;
        let q = vec![1.0; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 1, 1, 8, 4, 8)
            .await
            .unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_sparse_attention_large_batch() {
        let Some(dev) = get_test_device().await else {
            return;
        };
        // Batch size 4
        let size = 4 * 4 * 16 * 8;
        let q = vec![0.5; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &q, &q, 4, 4, 16, 8, 2)
            .await
            .unwrap();
        assert_eq!(output.len(), size);
    }

    #[tokio::test]
    async fn test_sparse_attention_precision() {
        let Some(dev) = get_test_device().await else {
            return;
        };
        // Verify attention output properties
        let size = 4 * 4;
        let q = vec![1.0; size];
        let k = vec![0.5; size];
        let v = vec![2.0; size];
        let output = sparse_attention(&dev.device, &dev.queue, &q, &k, &v, 1, 1, 4, 4, 2)
            .await
            .unwrap();

        assert_eq!(output.len(), size);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
