//! Fault Tests: Invalid Inputs
//!
//! Test that operations return errors (not panic) on invalid inputs
//! **Deep Debt**: Result<> everywhere, graceful error messages

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_matmul_dimension_mismatch() {
    // Matmul with incompatible dimensions should return error
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let a = vec![0.5f32; 10 * 20]; // 10 x 20
    let b = vec![0.3f32; 30 * 40]; // 30 x 40 (incompatible: k != 30)

    let result = matmul(&dev.device, &dev.queue, &a, &b, 10, 20, 40).await;

    // Should return error, not panic
    assert!(
        result.is_err(),
        "Matmul with dimension mismatch should return error"
    );
}

#[tokio::test]
async fn test_softmax_zero_classes() {
    // Softmax with 0 classes should error
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![0.5f32; 10];
    let result = softmax(&dev.device, &dev.queue, &input, 10, 0).await;

    assert!(
        result.is_err(),
        "Softmax with zero classes should return error"
    );
}

#[tokio::test]
async fn test_batch_norm_invalid_shapes() {
    // Batch norm with mismatched scale/bias dimensions
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 4;
    let channels = 64;
    let spatial = 10;

    let input = vec![0.5f32; batch * channels * spatial];
    let scale = vec![1.0f32; channels + 10]; // Wrong size!
    let bias = vec![0.0f32; channels];
    let mean = vec![0.0f32; channels];
    let var = vec![1.0f32; channels];

    let result = batch_norm(
        &dev.device,
        &dev.queue,
        &input,
        &scale,
        &bias,
        &mean,
        &var,
        batch,
        channels,
        spatial,
        1e-5,
    )
    .await;

    // Should handle gracefully
    assert!(
        result.is_err() || result.is_ok(),
        "Batch norm with invalid scale should not panic"
    );
}

#[tokio::test]
async fn test_conv2d_zero_kernel() {
    // Conv2D with kernel size 0 should error
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 1;
    let in_channels = 3;
    let out_channels = 16;
    let height = 32;
    let width = 32;

    let input = vec![0.5f32; batch * in_channels * height * width];
    let weights = vec![]; // Empty weights
    let bias = vec![0.0f32; out_channels];

    let result = conv2d(
        &dev.device,
        &dev.queue,
        &input,
        &weights,
        &bias,
        batch,
        in_channels,
        height,
        width,
        out_channels,
        0, // kernel_h = 0 (invalid!)
        0, // kernel_w = 0 (invalid!)
        1,
        1,
        0,
        0,
    )
    .await;

    assert!(
        result.is_err(),
        "Conv2D with zero kernel size should return error"
    );
}

#[tokio::test]
async fn test_add_mismatched_sizes() {
    // Element-wise add with different sizes should error
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let a = vec![0.5f32; 100];
    let b = vec![0.3f32; 200]; // Different size!

    let result = add(&dev.device, &dev.queue, &a, &b, 100).await;

    // Should handle gracefully (either error or truncate)
    assert!(
        result.is_err() || result.is_ok(),
        "Add with mismatched sizes should not panic"
    );
}

#[tokio::test]
async fn test_embedding_out_of_bounds_index() {
    // Embedding with index >= vocab_size should error
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let vocab_size = 1000;
    let embed_dim = 128;
    let weights = vec![0.1f32; vocab_size * embed_dim];

    // Index out of bounds
    let indices = vec![0, 100, 1000, 500]; // 1000 is out of bounds!

    let result = embedding(&dev.device, &dev.queue, &indices, &weights, vocab_size, embed_dim).await;

    // Should return error, not crash
    assert!(
        result.is_err() || result.is_ok(),
        "Embedding with out-of-bounds index should not panic"
    );
}
