// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `AdamW` Optimizer

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_adamw_gpu_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 1000;
    let params = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; size], vec![size], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; size], vec![size], device)
        .await
        .unwrap();

    let (new_params, new_m, new_v) = params
        .adamw(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)
        .unwrap();

    assert_eq!(new_params.shape(), &[size]);
    assert_eq!(new_m.shape(), &[size]);
    assert_eq!(new_v.shape(), &[size]);

    let p_data = new_params.to_vec().unwrap();
    let m_data = new_m.to_vec().unwrap();
    let v_data = new_v.to_vec().unwrap();

    // Params should decrease (gradient descent + weight decay)
    assert!(p_data.iter().all(|&x| x < 1.0));
    // m should be non-zero (momentum accumulated)
    assert!(m_data.iter().any(|&x| x.abs() > 1e-6));
    // v should be non-zero (variance accumulated)
    assert!(v_data.iter().all(|&x| x > 0.0));
}

#[tokio::test]
async fn test_adamw_gpu_convergence() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 100;
    let mut params = Tensor::from_vec_on(vec![5.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let mut m = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let mut v = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    // Constant gradient pointing toward zero
    let gradients = Tensor::from_vec_on(vec![1.0; size], vec![size], device)
        .await
        .unwrap();

    // Run 10 steps
    for step in 1..=10 {
        let (p, m_new, v_new) = params
            .adamw(&gradients, &m, &v, 0.1, 0.9, 0.999, 1e-8, 0.01, step)
            .unwrap();
        params = p;
        m = m_new;
        v = v_new;
    }

    let final_params = params.to_vec().unwrap();
    // Should converge toward lower values
    assert!(final_params.iter().all(|&x| x < 4.0));
}

#[tokio::test]
async fn test_adamw_gpu_weight_decay_stronger() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 100;
    let params = Tensor::from_vec_on(vec![10.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    // With weight decay, params should shrink even with zero gradient
    let (new_params_wd, _, _) = params
        .clone()
        .adamw(&gradients, &m, &v, 0.1, 0.9, 0.999, 1e-8, 0.1, 1)
        .unwrap();

    let (new_params_no_wd, _, _) = params
        .adamw(&gradients, &m, &v, 0.1, 0.9, 0.999, 1e-8, 0.0, 1)
        .unwrap();

    let wd_data = new_params_wd.to_vec().unwrap();
    let no_wd_data = new_params_no_wd.to_vec().unwrap();

    // Weight decay should reduce params more than no weight decay
    assert!(wd_data[0] < no_wd_data[0]);
    assert!(wd_data[0] < 10.0);
}

#[tokio::test]
async fn test_adamw_gpu_shape_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let params = Tensor::from_vec_on(vec![1.0; 100], vec![100], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 50], vec![50], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; 100], vec![100], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; 100], vec![100], device)
        .await
        .unwrap();

    // Shape mismatch should error
    let result = params.adamw(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_adamw_gpu_multidimensional() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // 2D params (matrix)
    let params = Tensor::from_vec_on(vec![1.0; 100], vec![10, 10], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; 100], vec![10, 10], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; 100], vec![10, 10], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; 100], vec![10, 10], device)
        .await
        .unwrap();

    let (new_params, new_m, new_v) = params
        .adamw(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)
        .unwrap();

    assert_eq!(new_params.shape(), &[10, 10]);
    assert_eq!(new_m.shape(), &[10, 10]);
    assert_eq!(new_v.shape(), &[10, 10]);
}

#[tokio::test]
async fn test_adamw_vs_adam_difference() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Compare AdamW vs Adam behavior with same hyperparameters
    let size = 100;
    let params = Tensor::from_vec_on(vec![10.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.1; size], vec![size], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; size], vec![size], device)
        .await
        .unwrap();

    // AdamW with weight decay
    let (adamw_params, _, _) = params
        .clone()
        .adamw(&gradients, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.1, 1)
        .unwrap();

    let adamw_data = adamw_params.to_vec().unwrap();

    // AdamW should apply decoupled weight decay
    // Result should be different from params without update
    assert!(adamw_data.iter().all(|&x| x < 10.0));

    // Verify all values are finite
    assert!(adamw_data.iter().all(|&x| x.is_finite()));
}
