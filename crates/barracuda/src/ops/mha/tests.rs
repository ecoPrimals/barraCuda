// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Multi-Head Attention
//!
//! Validates projection, attention, and cross-attention functionality.

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_mha_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 2;
    let seq_len = 8;
    let d_model = 64;
    let num_heads = 8;

    // Create inputs
    let q = Tensor::from_vec_on(
        vec![0.5; batch * seq_len * d_model],
        vec![batch, seq_len, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let k = Tensor::from_vec_on(
        vec![0.5; batch * seq_len * d_model],
        vec![batch, seq_len, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let v = Tensor::from_vec_on(
        vec![1.0; batch * seq_len * d_model],
        vec![batch, seq_len, d_model],
        device.clone(),
    )
    .await
    .unwrap();

    // Create projection weights
    let w_q = Tensor::from_vec_on(
        vec![0.01; d_model * d_model],
        vec![d_model, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let w_k = w_q.clone();
    let w_v = w_q.clone();
    let w_o = w_q.clone();

    let output = q
        .multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, num_heads)
        .unwrap();

    assert_eq!(output.shape(), &[batch, seq_len, d_model]);
}

#[tokio::test]
async fn test_mha_shape_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 2;
    let seq_len = 8;
    let d_model = 64;
    let num_heads = 8;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * seq_len * d_model],
        vec![batch, seq_len, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let w_q = Tensor::from_vec_on(
        vec![0.01; d_model * d_model],
        vec![d_model, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let w_k = w_q.clone();
    let w_v = w_q.clone();
    let w_o = w_q.clone();

    // Valid: d_model divisible by num_heads
    assert!(
        q.clone()
            .multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, num_heads)
            .is_ok()
    );

    // Invalid: d_model not divisible by num_heads
    let result = q.multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, 7);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_mha_cross_attention() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 2;
    let q_seq = 8;
    let kv_seq = 16; // Different sequence length for cross-attention
    let d_model = 64;
    let num_heads = 8;

    // Query has different seq_len than Key/Value (cross-attention)
    let q = Tensor::from_vec_on(
        vec![0.5; batch * q_seq * d_model],
        vec![batch, q_seq, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let k = Tensor::from_vec_on(
        vec![0.5; batch * kv_seq * d_model],
        vec![batch, kv_seq, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let v = Tensor::from_vec_on(
        vec![1.0; batch * kv_seq * d_model],
        vec![batch, kv_seq, d_model],
        device.clone(),
    )
    .await
    .unwrap();

    let w_q = Tensor::from_vec_on(
        vec![0.01; d_model * d_model],
        vec![d_model, d_model],
        device.clone(),
    )
    .await
    .unwrap();
    let w_k = w_q.clone();
    let w_v = w_q.clone();
    let w_o = w_q.clone();

    let output = q
        .multi_head_attention(&k, &v, &w_q, &w_k, &w_v, &w_o, num_heads)
        .unwrap();

    // Output shape matches query sequence length
    assert_eq!(output.shape(), &[batch, q_seq, d_model]);
}
