// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Sparse Attention
//!
//! Validates sparse attention with various stride values and sequence lengths.

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_sparse_attention_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let batch = 1;
    let heads = 2;
    let seq = 8;
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

    let output = q.sparse_attention(&k, &v, 2).unwrap(); // stride=2

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_sparse_attention_stride_1() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // stride=1 should work like full attention
    let batch = 1;
    let heads = 1;
    let seq = 4;
    let dim = 4;

    let q = Tensor::from_vec_on(
        vec![1.0; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let output = q.sparse_attention(&k, &v, 1).unwrap(); // stride=1 = full

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_sparse_attention_large_stride() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Large stride (attend to few positions)
    let batch = 2;
    let heads = 4;
    let seq = 16;
    let dim = 8;

    let q = Tensor::from_vec_on(
        vec![0.5; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let output = q.sparse_attention(&k, &v, 4).unwrap(); // stride=4

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_sparse_attention_long_sequence() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Long sequence (sparse is memory-efficient)
    let batch = 2;
    let heads = 8;
    let seq = 64; // longer sequence
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

    let output = q.sparse_attention(&k, &v, 8).unwrap(); // stride=8

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
}
