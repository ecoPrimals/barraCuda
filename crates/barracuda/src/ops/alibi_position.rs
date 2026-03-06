// SPDX-License-Identifier: AGPL-3.0-or-later
//! `ALiBi` Position Encoding - Attention with Linear Biases
//!
//! Adds position-dependent bias to attention scores.
//! No position embeddings needed - bias encodes position directly.
//!
//! Reference: "Train Short, Test Long" (Press et al., 2021)

use crate::error::Result;

/// `ALiBi` position encoding: adds position-dependent bias to attention scores.
///
/// No position embeddings needed — bias encodes position directly.
/// Head-specific slopes follow geometric sequence.
///
/// # Errors
///
/// Currently always succeeds; returns [`Err`] only if internal allocation fails.
pub async fn alibi_position(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    attention_scores: &[f32], // [batch, heads, seq_len, seq_len]
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
) -> Result<Vec<f32>> {
    let mut output = attention_scores.to_vec();

    // Head-specific slopes (geometric sequence)
    let slopes: Vec<f32> = (0..num_heads)
        .map(|h| 2.0_f32.powf(-(8.0 * (h + 1) as f32 / num_heads as f32)))
        .collect();

    for b in 0..batch_size {
        for h in 0..num_heads {
            let slope = slopes[h];

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = (i as isize - j as isize).abs() as f32;
                    let bias = -slope * distance;

                    let idx =
                        b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                    output[idx] += bias;
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
    async fn test_alibi_position_basic() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        let scores = vec![1.0; 2 * 4 * 4]; // batch=1, heads=2, seq=4
        let output = alibi_position(&dev.device, &dev.queue, &scores, 1, 2, 4)
            .await
            .unwrap();
        assert_eq!(output.len(), scores.len());
        assert!(output.iter().all(|&x| x.is_finite()));
        // Diagonal elements (distance=0) should equal original scores
        assert!((output[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_alibi_position_edge_cases() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with single head, single token
        let scores = vec![5.0];
        let output = alibi_position(&dev.device, &dev.queue, &scores, 1, 1, 1)
            .await
            .unwrap();
        assert_eq!(output.len(), 1);
        // Distance=0, no bias added
        assert!((output[0] - 5.0).abs() < 1e-6);

        // Test with zero attention scores
        let scores = vec![0.0; 4 * 4];
        let output = alibi_position(&dev.device, &dev.queue, &scores, 1, 1, 4)
            .await
            .unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_alibi_position_boundary() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with different numbers of heads (slope variations)
        let scores1 = vec![0.0; 4 * 4];
        let output1 = alibi_position(&dev.device, &dev.queue, &scores1, 1, 1, 4)
            .await
            .unwrap();

        let scores2 = vec![0.0; 4 * 4 * 4];
        let output2 = alibi_position(&dev.device, &dev.queue, &scores2, 1, 4, 4)
            .await
            .unwrap();

        assert!(output1.iter().all(|&x| x.is_finite()));
        assert!(output2.iter().all(|&x| x.is_finite()));
        // Different heads should have different biases
        assert_ne!(output1, scores1);
        assert_ne!(output2, scores2);
    }

    #[tokio::test]
    async fn test_alibi_position_large_batch() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Multiple batches, heads, longer sequences
        let batch_size = 4;
        let num_heads = 8;
        let seq_len = 16;

        let scores = vec![1.0; batch_size * num_heads * seq_len * seq_len];
        let output = alibi_position(
            &dev.device,
            &dev.queue,
            &scores,
            batch_size,
            num_heads,
            seq_len,
        )
        .await
        .unwrap();

        assert_eq!(output.len(), scores.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_alibi_position_precision() {
        let Some(dev) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with known distances and biases
        let scores = vec![0.0; 3 * 3]; // seq_len=3
        let output = alibi_position(&dev.device, &dev.queue, &scores, 1, 1, 3)
            .await
            .unwrap();

        // Slope for head 0: 2^(-8/1) = 2^-8 = 1/256
        let slope = 2.0_f32.powf(-8.0);

        // Position [0,0]: distance=0, bias=0
        assert!(output[0].abs() < 1e-6);

        // Position [0,1]: distance=1, bias=-slope*1
        assert!((output[1] + slope).abs() < 1e-6);

        // Position [0,2]: distance=2, bias=-slope*2
        assert!((output[2] + slope * 2.0).abs() < 1e-6);

        // Position [1,0]: distance=1, bias=-slope*1
        assert!((output[3] + slope).abs() < 1e-6);
    }
}
