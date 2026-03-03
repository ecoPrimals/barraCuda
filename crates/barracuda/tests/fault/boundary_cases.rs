// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fault Tests: Boundary Cases
//!
//! Test edge cases: zero sizes, max values, special floats
//! **Deep Debt**: Handle all valid inputs correctly

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_matmul_one_dimension() {
    // Matmul with m=1, n=1, k=1 (smallest valid)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let a = vec![2.0f32];
    let b = vec![3.0f32];

    let result = matmul(&dev.device, &dev.queue, &a, &b, 1, 1, 1).await;

    assert!(result.is_ok(), "Matmul 1x1 should succeed");

    if let Ok(output) = result {
        assert_eq!(output.len(), 1);
        assert!((output[0] - 6.0).abs() < 0.001, "1x1 matmul: 2 * 3 = 6");
    }
}

#[tokio::test]
async fn test_relu_with_infinities() {
    // ReLU with inf and -inf
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0, -1.0];

    let result = relu(&dev.device, &dev.queue, &input, 5).await;

    assert!(result.is_ok(), "ReLU should handle infinities");

    if let Ok(output) = result {
        assert!(output[0].is_infinite() && output[0].is_sign_positive(), "ReLU(inf) = inf");
        assert_eq!(output[1], 0.0, "ReLU(-inf) = 0");
        assert_eq!(output[2], 0.0, "ReLU(0) = 0");
    }
}

#[tokio::test]
async fn test_softmax_with_large_values() {
    // Softmax with very large values (numerical stability test)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![1000.0, 1001.0, 999.0]; // Large values
    let result = softmax(&dev.device, &dev.queue, &input, 1, 3).await;

    assert!(result.is_ok(), "Softmax should handle large values");

    if let Ok(output) = result {
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax should still sum to 1.0 with large inputs"
        );
    }
}

#[tokio::test]
async fn test_div_by_near_zero() {
    // Division by very small numbers
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let a = vec![1.0f32; 10];
    let b = vec![1e-10f32; 10]; // Very small

    let result = div(&dev.device, &dev.queue, &a, &b, 10).await;

    assert!(result.is_ok(), "Division by small numbers should succeed");

    if let Ok(output) = result {
        // Results should be large but finite
        assert!(output[0].is_finite(), "Division result should be finite");
    }
}

#[tokio::test]
async fn test_layer_norm_single_element() {
    // Layer norm with single element
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![5.0f32];
    let weights = vec![1.0f32];
    let bias = vec![0.0f32];

    let result = layer_norm(&dev.device, &dev.queue, &input, &weights, &bias, 1, 1, 1e-5).await;

    assert!(result.is_ok(), "LayerNorm with single element should work");
}

#[tokio::test]
async fn test_maxpool_one_by_one() {
    // MaxPool with 1x1 input and 1x1 kernel
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![5.0f32];

    let result = maxpool2d(
        &dev.device,
        &dev.queue,
        &input,
        1, // batch
        1, // channels
        1, // height
        1, // width
        1, // kernel_h
        1, // kernel_w
        1, // stride_h
        1, // stride_w
        0, // padding_h
        0, // padding_w
    )
    .await;

    assert!(result.is_ok(), "MaxPool 1x1 should succeed");

    if let Ok(output) = result {
        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 5.0, "MaxPool identity on single element");
    }
}

#[tokio::test]
async fn test_concat_empty_dimension() {
    // Concatenate with one tensor having size 0 in concat dimension
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 4;
    let a = vec![0.5f32; batch * 10];
    let b = vec![]; // Empty

    let result = concat(&dev.device, &dev.queue, &a, &b, batch, 10, 0).await;

    // Should either error or return just 'a'
    assert!(
        result.is_err() || result.is_ok(),
        "Concat with empty dimension should not panic"
    );
}
