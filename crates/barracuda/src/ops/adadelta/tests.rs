// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `AdaDelta` Optimizer

use super::*;
use crate::device::test_pool::get_test_device;

#[tokio::test]
async fn test_adadelta_basic() {
    let device = get_test_device().await;
    let weights = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
        .await
        .unwrap();

    let (updated_weights, _acc_grad, _acc_delta) =
        weights.adadelta_step(&gradients, 0.95, None, None).unwrap();
    let result = updated_weights.to_vec().unwrap();

    // Weights should be updated
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&x| x.is_finite()));
    // AdaDelta should decrease weights (gradient descent)
    assert!(
        result[0] < 1.0,
        "Expected result[0] < 1.0, got {}",
        result[0]
    );
}

#[tokio::test]
async fn test_adadelta_zero_gradients() {
    let device = get_test_device().await;
    let weights = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.0, 0.0], vec![2], device.clone())
        .await
        .unwrap();

    let (updated_weights, acc_grad, acc_delta) =
        weights.adadelta_step(&gradients, 0.95, None, None).unwrap();
    let result = updated_weights.to_vec().unwrap();
    let ag = acc_grad.to_vec().unwrap();
    let ad = acc_delta.to_vec().unwrap();

    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(ag.iter().all(|&x| x.is_finite()));
    assert!(ad.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_adadelta_different_rho() {
    let device = get_test_device().await;
    let weights1 = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let weights2 = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    // Low rho (less momentum)
    let (updated1, _ag, _ad) = weights1.adadelta_step(&gradients, 0.5, None, None).unwrap();

    // High rho (more momentum)
    let (updated2, _ag, _ad) = weights2
        .adadelta_step(&gradients, 0.99, None, None)
        .unwrap();

    let result1 = updated1.to_vec().unwrap();
    let result2 = updated2.to_vec().unwrap();

    assert!(result1.iter().all(|&x| x.is_finite()));
    assert!(result2.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_adadelta_validation() {
    let device = get_test_device().await;
    let weights = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
        .await
        .unwrap();
    let gradients = Tensor::from_vec_on(vec![0.1; 5], vec![5], device.clone())
        .await
        .unwrap();

    // Shape mismatch
    assert!(
        weights
            .clone()
            .adadelta_step(&gradients, 0.95, None, None)
            .is_err()
    );

    // Invalid rho
    let gradients_correct = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
        .await
        .unwrap();
    assert!(
        weights
            .clone()
            .adadelta_step(&gradients_correct, -0.1, None, None)
            .is_err()
    );
    assert!(
        weights
            .clone()
            .adadelta_step(&gradients_correct, 1.5, None, None)
            .is_err()
    );
}

#[tokio::test]
async fn test_adadelta_large_batch() {
    let device = get_test_device().await;
    let size = 128;
    let weights_data: Vec<f32> = (0..size).map(|i| (i as f32) / 10.0).collect();
    let grads_data = vec![0.01; size];

    let weights = Tensor::from_vec_on(weights_data, vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(grads_data, vec![size], device.clone())
        .await
        .unwrap();

    let (updated_weights, updated_ag, updated_ad) =
        weights.adadelta_step(&gradients, 0.95, None, None).unwrap();

    let result = updated_weights.to_vec().unwrap();
    let ag = updated_ag.to_vec().unwrap();
    let ad = updated_ad.to_vec().unwrap();

    assert_eq!(result.len(), size);
    assert_eq!(ag.len(), size);
    assert_eq!(ad.len(), size);
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_adadelta_multi_step() {
    let device = get_test_device().await;
    let weights = Tensor::from_vec_on(vec![10.0, 20.0], vec![2], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
        .await
        .unwrap();

    // Step 1
    let (weights1, ag1, ad1) = weights.adadelta_step(&gradients, 0.95, None, None).unwrap();
    let result1 = weights1.to_vec().unwrap();

    assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
    assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

    // Step 2 with accumulated state
    let (weights2, _ag2, _ad2) = weights1
        .adadelta_step(&gradients, 0.95, Some(&ag1), Some(&ad1))
        .unwrap();
    let result2 = weights2.to_vec().unwrap();

    // Should continue optimizing
    assert!(result2.iter().all(|&x| x.is_finite()));
    assert!(result2[0] < 10.0);
    assert!(result2[1] < 20.0);
}
