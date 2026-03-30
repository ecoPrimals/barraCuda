// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for Variance operation

use super::*;
use crate::device::test_pool::get_test_device;

fn variance_cpu(input: &[f32]) -> f32 {
    let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
    variance
}

#[tokio::test]
async fn test_variance_basic() {
    let device = get_test_device().await;
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
        .await
        .unwrap();
    let result = input.variance().unwrap().to_vec().unwrap();
    let expected = variance_cpu(&input_data);

    assert!(
        (result[0] - expected).abs() < 1e-4,
        "Expected {}, got {}",
        expected,
        result[0]
    );
}

#[tokio::test]
async fn test_variance_edge_cases() {
    let device = get_test_device().await;
    // All zeros (variance = 0)
    let input_data = vec![0.0, 0.0, 0.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
        .await
        .unwrap();
    let result = input.variance().unwrap().to_vec().unwrap();
    assert!(result[0].abs() < 1e-6);

    // All same value (variance = 0)
    let input_data = vec![5.0, 5.0, 5.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
        .await
        .unwrap();
    let result = input.variance().unwrap().to_vec().unwrap();
    assert!(result[0].abs() < 1e-6);
}

#[tokio::test]
async fn test_variance_boundary() {
    let device = get_test_device().await;
    let input_data = vec![0.0, 10.0, 20.0, 30.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
        .await
        .unwrap();
    let result = input.variance().unwrap().to_vec().unwrap();
    let expected = variance_cpu(&input_data);

    let rel_error = if expected > 1e-5 {
        (result[0] - expected).abs() / expected
    } else {
        (result[0] - expected).abs()
    };
    assert!(rel_error < 1e-3, "Expected {}, got {}", expected, result[0]);
}

#[tokio::test]
async fn test_variance_large_tensor() {
    let device = get_test_device().await;
    let size = 100;
    let input_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
    let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
        .await
        .unwrap();
    let result = input.variance().unwrap().to_vec().unwrap();
    let expected = variance_cpu(&input_data);

    let rel_error = (result[0] - expected).abs() / expected;
    assert!(rel_error < 1e-2, "Expected {}, got {}", expected, result[0]);
}

#[tokio::test]
async fn test_variance_precision() {
    let device = get_test_device().await;
    let input_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
        .await
        .unwrap();
    let gpu_result = input.variance().unwrap().to_vec().unwrap();
    let cpu_result = variance_cpu(&input_data);

    let error = (gpu_result[0] - cpu_result).abs();
    assert!(error < 1e-3, "Error {error} exceeds threshold");
}

#[tokio::test]
async fn test_variance_dim() {
    let device = get_test_device().await;
    // Test 2D tensor: [[1, 2, 3], [4, 5, 6]]
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
        .await
        .unwrap();

    // Variance along dim 0 (columns): variance of [1,4], [2,5], [3,6]
    let result = input.variance_dim(0, false).unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 3);
    // Variance of [1, 4] = ((1-2.5)^2 + (4-2.5)^2) / 2 = (2.25 + 2.25) / 2 = 2.25
    assert!((result[0] - 2.25).abs() < 1e-4);
    // Variance of [2, 5] = ((2-3.5)^2 + (5-3.5)^2) / 2 = (2.25 + 2.25) / 2 = 2.25
    assert!((result[1] - 2.25).abs() < 1e-4);
    // Variance of [3, 6] = ((3-4.5)^2 + (6-4.5)^2) / 2 = (2.25 + 2.25) / 2 = 2.25
    assert!((result[2] - 2.25).abs() < 1e-4);

    // Variance along dim 1 (rows): variance of [1,2,3], [4,5,6]
    let result = input.variance_dim(1, false).unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 2);
    // Variance of [1, 2, 3] = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
    assert!((result[0] - 0.6666667).abs() < 1e-4);
    // Variance of [4, 5, 6] = ((4-5)^2 + (5-5)^2 + (6-5)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
    assert!((result[1] - 0.6666667).abs() < 1e-4);

    // Variance along dim 0 with keepdim: [[2.25, 2.25, 2.25]]
    let result = input.variance_dim(0, true).unwrap();
    assert_eq!(result.shape(), &[1, 3]);
}
