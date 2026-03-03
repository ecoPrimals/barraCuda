// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Causal Attention
//!
//! Validates GPT-style autoregressive attention with causal masking.

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_causal_attention_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 1;
    let heads = 2;
    let seq = 8;
    let dim = 16;

    // Create inputs
    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = Tensor::from_vec_on(
        vec![0.5; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let v = Tensor::from_vec_on(
        vec![1.0; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device,
    )
    .await
    .unwrap();

    // Execute
    let output = q.causal_attention(&k, &v).unwrap();

    // Validate shape
    assert_eq!(output.shape(), &[batch, heads, seq, dim]);

    // Validate values are finite
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_causal_attention_single_token() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 1;
    let heads = 1;
    let seq = 1; // Single token - no masking needed
    let dim = 4;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let output = q.causal_attention(&k, &v).unwrap();

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_causal_attention_gpt_style() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // GPT-style dimensions
    let batch = 2;
    let heads = 8;
    let seq = 16;
    let dim = 16;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let output = q.causal_attention(&k, &v).unwrap();

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}
