// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Triplet Loss operation

use super::*;
use crate::device::test_pool::get_test_device;

#[tokio::test]
async fn test_triplet_loss_gpu_basic() {
    let device = get_test_device().await;
    let batch = 32;
    let embedding_dim = 128;

    // Create triplets
    let anchors = Tensor::from_vec_on(
        vec![1.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let positives = Tensor::from_vec_on(
        vec![1.1; batch * embedding_dim], // Close to anchors
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let negatives = Tensor::from_vec_on(
        vec![5.0; batch * embedding_dim], // Far from anchors
        vec![batch, embedding_dim],
        device,
    )
    .await
    .unwrap();

    let loss = anchors.triplet_loss(&positives, &negatives, 0.2).unwrap();

    assert_eq!(loss.shape(), &[batch]);
    let data = loss.to_vec().unwrap();

    // Loss should be low (negatives are far enough)
    assert!(data.iter().all(|&x| x >= 0.0 && x.is_finite()));
}

#[tokio::test]
async fn test_triplet_loss_gpu_hard_negative() {
    let device = get_test_device().await;
    let batch = 16;
    let embedding_dim = 64;

    // All embeddings very similar (hard negative case)
    let anchors = Tensor::from_vec_on(
        vec![1.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let positives = Tensor::from_vec_on(
        vec![1.02; batch * embedding_dim], // Very close
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let negatives = Tensor::from_vec_on(
        vec![1.03; batch * embedding_dim], // Also very close (hard negative!)
        vec![batch, embedding_dim],
        device,
    )
    .await
    .unwrap();

    let loss = anchors.triplet_loss(&positives, &negatives, 0.2).unwrap();
    let data = loss.to_vec().unwrap();

    // Loss should be positive (negatives not far enough from positives)
    assert!(data.iter().all(|&x| x > 0.0));
}

#[tokio::test]
async fn test_triplet_loss_gpu_easy_negative() {
    let device = get_test_device().await;
    let batch = 8;
    let embedding_dim = 32;

    let anchors = Tensor::from_vec_on(
        vec![0.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let positives = Tensor::from_vec_on(
        vec![0.1; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let negatives = Tensor::from_vec_on(
        vec![10.0; batch * embedding_dim], // Very far (easy negative)
        vec![batch, embedding_dim],
        device,
    )
    .await
    .unwrap();

    let loss = anchors.triplet_loss(&positives, &negatives, 0.2).unwrap();
    let data = loss.to_vec().unwrap();

    // Loss should be zero or near-zero (negatives far enough)
    assert!(data.iter().all(|&x| x < 0.1));
}

#[tokio::test]
async fn test_triplet_loss_gpu_cosine_distance() {
    let device = get_test_device().await;
    let batch = 16;
    let embedding_dim = 128;

    let anchors = Tensor::from_vec_on(
        vec![1.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let positives = Tensor::from_vec_on(
        vec![0.9; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let negatives = Tensor::from_vec_on(
        vec![-1.0; batch * embedding_dim], // Opposite direction
        vec![batch, embedding_dim],
        device,
    )
    .await
    .unwrap();

    let loss = anchors
        .triplet_loss_cosine(&positives, &negatives, 0.1)
        .unwrap();

    assert_eq!(loss.shape(), &[batch]);
    let data = loss.to_vec().unwrap();
    assert!(data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_triplet_loss_gpu_margin_effect() {
    let device = get_test_device().await;
    let batch = 8;
    let embedding_dim = 32;

    let anchors = Tensor::from_vec_on(
        vec![1.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let positives = Tensor::from_vec_on(
        vec![1.2; batch * embedding_dim],
        vec![batch, embedding_dim],
        device.clone(),
    )
    .await
    .unwrap();

    let negatives = Tensor::from_vec_on(
        vec![2.0; batch * embedding_dim],
        vec![batch, embedding_dim],
        device,
    )
    .await
    .unwrap();

    // Small margin
    let loss_small = anchors
        .clone()
        .triplet_loss(&positives, &negatives, 0.1)
        .unwrap();

    // Large margin
    let loss_large = anchors.triplet_loss(&positives, &negatives, 1.0).unwrap();

    let data_small = loss_small.to_vec().unwrap();
    let data_large = loss_large.to_vec().unwrap();

    // Larger margin should result in higher loss
    assert!(data_large[0] >= data_small[0]);
}

#[tokio::test]
async fn test_triplet_loss_gpu_validation() {
    let device = get_test_device().await;
    let anchors = Tensor::from_vec_on(vec![1.0; 100], vec![10, 10], device.clone())
        .await
        .unwrap();

    let positives = Tensor::from_vec_on(vec![1.0; 50], vec![10, 5], device.clone())
        .await
        .unwrap();

    let negatives = Tensor::from_vec_on(vec![1.0; 100], vec![10, 10], device)
        .await
        .unwrap();

    // Shape mismatch should error
    assert!(anchors.triplet_loss(&positives, &negatives, 0.2).is_err());
}
