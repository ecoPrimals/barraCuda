// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Adam Optimizer

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_adam_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
        .await
        .unwrap();

    let (updated_params, _m, _v) = params
        .adam_step(&gradients, 0.001, 0.9, 0.999, 1, None, None)
        .unwrap();
    let result = updated_params.to_vec().unwrap();

    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|&x| x.is_finite()));
    assert!(result[0] < 1.0, "Expected descent, got {}", result[0]);
}

#[tokio::test]
async fn test_adam_with_state() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
        .await
        .unwrap();

    // Step 1
    let (params1, m1, v1) = params
        .adam_step(&gradients, 0.001, 0.9, 0.999, 1, None, None)
        .unwrap();

    let m_data = m1.to_vec().unwrap();
    let v_data = v1.to_vec().unwrap();
    assert!(m_data.iter().all(|&x| x.is_finite()));
    assert!(v_data.iter().all(|&x| x.is_finite()));

    // Step 2 with accumulated state
    let (params2, _m2, _v2) = params1
        .adam_step(&gradients, 0.001, 0.9, 0.999, 2, Some(&m1), Some(&v1))
        .unwrap();

    let result = params2.to_vec().unwrap();
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_adam_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let params = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
        .await
        .unwrap();
    let gradients = Tensor::from_vec_on(vec![0.1; 5], vec![5], device.clone())
        .await
        .unwrap();
    let grads_correct = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
        .await
        .unwrap();

    // Shape mismatch
    assert!(params
        .clone()
        .adam_step(&gradients, 0.001, 0.9, 0.999, 1, None, None)
        .is_err());

    // Invalid learning rate
    assert!(params
        .clone()
        .adam_step(&grads_correct, -0.001, 0.9, 0.999, 1, None, None)
        .is_err());

    // Invalid beta1
    assert!(params
        .clone()
        .adam_step(&grads_correct, 0.001, -0.1, 0.999, 1, None, None)
        .is_err());
    assert!(params
        .clone()
        .adam_step(&grads_correct, 0.001, 1.0, 0.999, 1, None, None)
        .is_err());

    // Invalid step
    assert!(params
        .adam_step(&grads_correct, 0.001, 0.9, 0.999, 0, None, None)
        .is_err());
}

#[tokio::test]
async fn test_adam_large_batch() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 128;
    let params = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
        .await
        .unwrap();

    let (updated_params, updated_m, updated_v) = params
        .adam_step(&gradients, 0.001, 0.9, 0.999, 1, None, None)
        .unwrap();

    let result = updated_params.to_vec().unwrap();
    let m = updated_m.to_vec().unwrap();
    let v = updated_v.to_vec().unwrap();

    assert_eq!(result.len(), size);
    assert_eq!(m.len(), size);
    assert_eq!(v.len(), size);
    assert!(result.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_adam_multi_step() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let params = Tensor::from_vec_on(vec![10.0, 20.0], vec![2], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
        .await
        .unwrap();

    // Step 1
    let (params1, m1, v1) = params
        .adam_step(&gradients, 0.01, 0.9, 0.999, 1, None, None)
        .unwrap();
    let result1 = params1.to_vec().unwrap();

    assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
    assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

    // Step 2 with accumulated state
    let (params2, _m2, _v2) = params1
        .adam_step(&gradients, 0.01, 0.9, 0.999, 2, Some(&m1), Some(&v1))
        .unwrap();
    let result2 = params2.to_vec().unwrap();

    // Should continue descending
    assert!(result2[0] < result1[0]);
    assert!(result2[1] < result1[1]);
}
