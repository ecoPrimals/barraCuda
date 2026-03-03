//! Tests for NAdam Optimizer

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_nadam_gpu_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 1000;
    let weights = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
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

    let (new_weights, new_m, new_v) = weights
        .nadam(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1)
        .unwrap();

    assert_eq!(new_weights.shape(), &[size]);
    assert_eq!(new_m.shape(), &[size]);
    assert_eq!(new_v.shape(), &[size]);

    let w_data = new_weights.to_vec().unwrap();
    let m_data = new_m.to_vec().unwrap();
    let v_data = new_v.to_vec().unwrap();

    // Weights should decrease (gradient descent)
    assert!(w_data.iter().all(|&x| x < 1.0));
    // m should be non-zero (momentum accumulated)
    assert!(m_data.iter().any(|&x| x.abs() > 1e-6));
    // v should be non-zero (variance accumulated)
    assert!(v_data.iter().all(|&x| x > 0.0));
}

#[tokio::test]
async fn test_nadam_gpu_convergence() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 100;
    let mut weights = Tensor::from_vec_on(vec![5.0; size], vec![size], device.clone())
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
        let (w, m_new, v_new) = weights
            .nadam(&gradients, &m, &v, 0.1, 0.9, 0.999, 1e-8, 0.0, step)
            .unwrap();
        weights = w;
        m = m_new;
        v = v_new;
    }

    let final_weights = weights.to_vec().unwrap();
    // Should converge toward lower values
    assert!(final_weights.iter().all(|&x| x < 4.0));
}

#[tokio::test]
async fn test_nadam_gpu_weight_decay() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 100;
    let weights = Tensor::from_vec_on(vec![10.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let gradients = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let m = Tensor::from_vec_on(vec![0.0; size], vec![size], device.clone())
        .await
        .unwrap();

    let v = Tensor::from_vec_on(vec![0.0; size], vec![size], device)
        .await
        .unwrap();

    // With weight decay, weights should shrink even with zero gradient
    let (new_weights, _, _) = weights
        .nadam(&gradients, &m, &v, 0.1, 0.9, 0.999, 1e-8, 0.01, 1)
        .unwrap();

    let w_data = new_weights.to_vec().unwrap();
    // Weight decay should reduce weights
    assert!(w_data.iter().all(|&x| x < 10.0));
}

#[tokio::test]
async fn test_nadam_gpu_shape_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let weights = Tensor::from_vec_on(vec![1.0; 100], vec![100], device.clone())
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
    let result = weights.nadam(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_nadam_gpu_multidimensional() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // 2D weights (matrix)
    let weights = Tensor::from_vec_on(vec![1.0; 100], vec![10, 10], device.clone())
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

    let (new_weights, new_m, new_v) = weights
        .nadam(&gradients, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1)
        .unwrap();

    assert_eq!(new_weights.shape(), &[10, 10]);
    assert_eq!(new_m.shape(), &[10, 10]);
    assert_eq!(new_v.shape(), &[10, 10]);
}
