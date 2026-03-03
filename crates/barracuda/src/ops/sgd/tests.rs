// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for SGD Optimizer

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_sgd_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
        .await
        .unwrap();

    let (updated_weights, _) = weights.sgd_step(&gradients, 0.01, 0.0, 0.0, None).unwrap();
    let result = updated_weights.to_vec().unwrap();

    // Weights should decrease (gradient descent)
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(result[0] < 1.0, "Expected descent, got {}", result[0]);
}

#[tokio::test]
async fn test_sgd_with_momentum() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    // First step with momentum
    let (weights1, velocity1) = weights.sgd_step(&gradients, 0.01, 0.9, 0.0, None).unwrap();

    assert!(velocity1.is_some());
    let v = velocity1.unwrap();
    let v_data = v.to_vec().unwrap();
    assert!(v_data.iter().all(|&x| x.is_finite()));

    // Second step with accumulated momentum
    let (weights2, _velocity2) = weights1
        .sgd_step(&gradients, 0.01, 0.9, 0.0, Some(&v))
        .unwrap();

    let result = weights2.to_vec().unwrap();
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_sgd_with_weight_decay() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    let (updated_weights, _) = weights
        .sgd_step(&gradients, 0.01, 0.0, 0.001, None)
        .unwrap();

    let result = updated_weights.to_vec().unwrap();
    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(result[0] < 1.0); // Should have decreased
}

#[tokio::test]
async fn test_sgd_validation() {
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
        .sgd_step(&gradients, 0.01, 0.0, 0.0, None)
        .is_err());

    // Invalid learning rate
    assert!(weights
        .clone()
        .sgd_step(&grads_correct, -0.01, 0.0, 0.0, None)
        .is_err());
    assert!(weights
        .clone()
        .sgd_step(&grads_correct, 0.0, 0.0, 0.0, None)
        .is_err());

    // Invalid momentum
    assert!(weights
        .clone()
        .sgd_step(&grads_correct, 0.01, -0.1, 0.0, None)
        .is_err());
    assert!(weights
        .clone()
        .sgd_step(&grads_correct, 0.01, 1.5, 0.0, None)
        .is_err());

    // Invalid weight decay
    assert!(weights
        .sgd_step(&grads_correct, 0.01, 0.0, -0.001, None)
        .is_err());
}

#[tokio::test]
async fn test_sgd_large_batch() {
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

    let (updated_weights, _) = weights.sgd_step(&gradients, 0.01, 0.0, 0.0, None).unwrap();

    let result = updated_weights.to_vec().unwrap();
    assert_eq!(result.len(), size);
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_sgd_multi_step() {
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
    let (weights1, v1) = weights.sgd_step(&gradients, 0.1, 0.9, 0.0, None).unwrap();
    let result1 = weights1.to_vec().unwrap();

    assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
    assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

    // Step 2 with momentum
    let (weights2, _v2) = weights1
        .sgd_step(&gradients, 0.1, 0.9, 0.0, v1.as_ref())
        .unwrap();
    let result2 = weights2.to_vec().unwrap();

    // Should continue descending
    assert!(result2[0] < result1[0]);
    assert!(result2[1] < result1[1]);
}
