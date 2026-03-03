// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for RMSprop Optimizer

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_rmsprop_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
        .await
        .unwrap();

    let (updated_weights, updated_sq_avg) =
        weights.rmsprop_step(&gradients, 0.001, 0.99, None).unwrap();

    let result = updated_weights.to_vec().unwrap();
    let sq_avg = updated_sq_avg.to_vec().unwrap();

    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(sq_avg.iter().all(|&x| x.is_finite()));
    assert!(result[0] < 1.0, "Expected descent, got {}", result[0]);
}

#[tokio::test]
async fn test_rmsprop_accumulation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    // First step
    let (weights1, sq_avg1) = weights.rmsprop_step(&gradients, 0.001, 0.99, None).unwrap();

    let sq1 = sq_avg1.to_vec().unwrap();
    assert!(sq1.iter().all(|&x| x >= 0.0));

    // Second step with accumulated state
    let (weights2, sq_avg2) = weights1
        .rmsprop_step(&gradients, 0.001, 0.99, Some(&sq_avg1))
        .unwrap();

    let result = weights2.to_vec().unwrap();
    let sq2 = sq_avg2.to_vec().unwrap();

    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(sq2.iter().all(|&x| x >= sq1[0])); // Should accumulate
}

#[tokio::test]
async fn test_rmsprop_different_alpha() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights1 = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let weights2 = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    // Low alpha (less history)
    let (updated1, _) = weights1.rmsprop_step(&gradients, 0.001, 0.5, None).unwrap();

    // High alpha (more history)
    let (updated2, _) = weights2
        .rmsprop_step(&gradients, 0.001, 0.99, None)
        .unwrap();

    let result1 = updated1.to_vec().unwrap();
    let result2 = updated2.to_vec().unwrap();

    assert!(result1.iter().all(|&x| x.is_finite()));
    assert!(result2.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_rmsprop_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
        .await
        .unwrap();
    let gradients = Tensor::from_vec_on(vec![0.1; 5], vec![5], device.clone())
        .await
        .unwrap();
    let grads_correct = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
        .await
        .unwrap();

    // Shape mismatch
    assert!(weights
        .clone()
        .rmsprop_step(&gradients, 0.001, 0.99, None)
        .is_err());

    // Invalid learning rate
    assert!(weights
        .clone()
        .rmsprop_step(&grads_correct, -0.001, 0.99, None)
        .is_err());
    assert!(weights
        .clone()
        .rmsprop_step(&grads_correct, 0.0, 0.99, None)
        .is_err());

    // Invalid alpha
    assert!(weights
        .clone()
        .rmsprop_step(&grads_correct, 0.001, -0.1, None)
        .is_err());
    assert!(weights
        .clone()
        .rmsprop_step(&grads_correct, 0.001, 1.5, None)
        .is_err());
}

#[tokio::test]
async fn test_rmsprop_large_batch() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 128;
    let weights = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
        .await
        .unwrap();

    let (updated_weights, updated_sq_avg) =
        weights.rmsprop_step(&gradients, 0.001, 0.99, None).unwrap();

    let result = updated_weights.to_vec().unwrap();
    let sq_avg = updated_sq_avg.to_vec().unwrap();

    assert_eq!(result.len(), size);
    assert_eq!(sq_avg.len(), size);
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_rmsprop_multi_step() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![10.0, 20.0], vec![2], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
        .await
        .unwrap();

    // Step 1
    let (weights1, sq1) = weights.rmsprop_step(&gradients, 0.01, 0.99, None).unwrap();
    let result1 = weights1.to_vec().unwrap();

    assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
    assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

    // Step 2 with accumulated state
    let (weights2, _sq2) = weights1
        .rmsprop_step(&gradients, 0.01, 0.99, Some(&sq1))
        .unwrap();
    let result2 = weights2.to_vec().unwrap();

    // Should continue descending
    assert!(result2[0] < result1[0]);
    assert!(result2[1] < result1[1]);
}
