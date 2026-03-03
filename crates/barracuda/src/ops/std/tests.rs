//! Tests for Standard Deviation operation

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

fn std_cpu(input: &[f32]) -> f32 {
    let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
    let variance: f32 = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input.len() as f32;
    variance.sqrt()
}

#[tokio::test]
async fn test_std_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
        .await
        .unwrap();
    let result = input.std().unwrap().to_vec().unwrap();
    let expected = std_cpu(&input_data);

    assert!(
        (result[0] - expected).abs() < 1e-4,
        "Expected {}, got {}",
        expected,
        result[0]
    );
}

#[tokio::test]
async fn test_std_edge_cases() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // All same value (std = 0)
    let input_data = vec![5.0, 5.0, 5.0, 5.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
        .await
        .unwrap();
    let result = input.std().unwrap().to_vec().unwrap();
    assert!(result[0].abs() < 1e-6);

    // All zeros (std = 0)
    let input_data = vec![0.0, 0.0, 0.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
        .await
        .unwrap();
    let result = input.std().unwrap().to_vec().unwrap();
    assert!(result[0].abs() < 1e-6);
}

#[tokio::test]
async fn test_std_boundary() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input_data = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
        .await
        .unwrap();
    let result = input.std().unwrap().to_vec().unwrap();
    let expected = std_cpu(&input_data);

    let rel_error = if expected > 1e-5 {
        (result[0] - expected).abs() / expected
    } else {
        (result[0] - expected).abs()
    };
    assert!(rel_error < 1e-2, "Expected {}, got {}", expected, result[0]);
}

#[tokio::test]
async fn test_std_large_tensor() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let size = 100;
    let input_data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
    let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
        .await
        .unwrap();
    let result = input.std().unwrap().to_vec().unwrap();
    let expected = std_cpu(&input_data);

    let rel_error = (result[0] - expected).abs() / expected;
    assert!(rel_error < 1e-2, "Expected {}, got {}", expected, result[0]);
}

#[tokio::test]
async fn test_std_precision() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input_data = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![6], device)
        .await
        .unwrap();
    let gpu_result = input.std().unwrap().to_vec().unwrap();
    let cpu_result = std_cpu(&input_data);

    let error = (gpu_result[0] - cpu_result).abs();
    assert!(error < 1e-3, "Error {} exceeds threshold", error);
}

#[tokio::test]
async fn test_std_dim() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Test 2D tensor: [[1, 2, 3], [4, 5, 6]]
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
        .await
        .unwrap();

    // Std along dim 0 (columns): std of [1,4], [2,5], [3,6]
    let result = input.std_dim(0, false).unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 3);
    // Std of [1, 4] = sqrt(2.25) = 1.5
    assert!((result[0] - 1.5).abs() < 1e-4);
    // Std of [2, 5] = sqrt(2.25) = 1.5
    assert!((result[1] - 1.5).abs() < 1e-4);
    // Std of [3, 6] = sqrt(2.25) = 1.5
    assert!((result[2] - 1.5).abs() < 1e-4);

    // Std along dim 1 (rows): std of [1,2,3], [4,5,6]
    let result = input.std_dim(1, false).unwrap().to_vec().unwrap();
    assert_eq!(result.len(), 2);
    // Std of [1, 2, 3] = sqrt(0.666...) ≈ 0.8165
    assert!((result[0] - 0.8164966).abs() < 1e-4);
    // Std of [4, 5, 6] = sqrt(0.666...) ≈ 0.8165
    assert!((result[1] - 0.8164966).abs() < 1e-4);

    // Std along dim 0 with keepdim: [[1.5, 1.5, 1.5]]
    let result = input.std_dim(0, true).unwrap();
    assert_eq!(result.shape(), &[1, 3]);
}
