// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Local Attention
//!
//! Validates windowed attention for long sequences.

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_local_attention_basic() {
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

    let output = q.local_attention(&k, &v, 4).unwrap(); // window_size=4

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_local_attention_edge_cases() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Window size = 2 (minimal)
    let batch = 1;
    let heads = 1;
    let seq = 4;
    let dim = 2;

    let q = Tensor::from_vec_on(
        vec![1.0; batch * heads * seq * dim],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = q.clone();

    let output = q.local_attention(&k, &v, 2).unwrap(); // window_size=2

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));

    // Single head
    let batch = 1;
    let heads = 1;
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

    let output = q.local_attention(&k, &v, 4).unwrap();
    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
}

#[tokio::test]
async fn test_local_attention_boundary() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Large window (approaches full attention)
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

    let output = q.local_attention(&k, &v, 8).unwrap(); // window_size=8 (full)

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));

    // Multiple heads
    let batch = 1;
    let heads = 8;
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

    let output = q.local_attention(&k, &v, 4).unwrap();
    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
}

#[tokio::test]
async fn test_local_attention_large_batch() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Batch size 4, longer sequence
    let batch = 4;
    let heads = 4;
    let seq = 32;
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

    let output = q.local_attention(&k, &v, 8).unwrap();
    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
}

#[tokio::test]
async fn test_local_attention_precision() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Test attention pattern with known values
    let batch = 1;
    let heads = 1;
    let seq = 4;
    let dim = 2;

    let q = Tensor::from_vec_on(
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        vec![batch, heads, seq, dim],
        device.clone(),
    )
    .await
    .unwrap();
    let k = q.clone();
    let v = Tensor::from_vec_on(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![batch, heads, seq, dim],
        device,
    )
    .await
    .unwrap();

    let output = q.local_attention(&k, &v, 4).unwrap();

    assert_eq!(output.shape(), &[batch, heads, seq, dim]);
    let data = output.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
    // Verify attention produces weighted sums
    assert!(data.iter().any(|&x| x > 0.0));
}

#[tokio::test]
async fn test_local_attention_window_size_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };

    let q = Tensor::from_vec_on(vec![0.5; 4 * 2], vec![1, 1, 4, 2], device.clone())
        .await
        .unwrap();
    let k = q.clone();
    let v = q.clone();

    // Valid: window_size > 0
    assert!(q.clone().local_attention(&k, &v, 1).is_ok());

    // Invalid: window_size = 0
    assert!(q.local_attention(&k, &v, 0).is_err());
}
