// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chaos Tests: Stress Testing
//!
//! Test operations under memory pressure and large inputs
//! **Deep Debt**: Should handle large workloads gracefully

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_large_matmul() {
    // Large matrix multiplication (1024 x 1024)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let m = 1024;
    let n = 1024;
    let k = 1024;

    let a = vec![0.5f32; m * k];
    let b = vec![0.3f32; k * n];

    let result = matmul(&dev.device, &dev.queue, &a, &b, m, k, n).await;

    assert!(result.is_ok(), "Should handle large matmul 1024x1024");

    if let Ok(output) = result {
        assert_eq!(output.len(), m * n);
    }
}

#[tokio::test]
async fn test_large_batch_normalization() {
    // Large batch norm (1000 samples, 512 features)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 1000;
    let channels = 512;
    let spatial = 1;

    let input = vec![0.5f32; batch * channels * spatial];
    let scale = vec![1.0f32; channels];
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

    assert!(result.is_ok(), "Should handle large batch norm");
}

#[tokio::test]
async fn test_many_small_operations() {
    // Run 1000 small operations in sequence
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 100;
    let num_ops = 1000;

    let mut data = vec![0.5f32; size];

    for i in 0..num_ops {
        // Alternate between different operations
        data = match i % 4 {
            0 => relu(&dev.device, &dev.queue, &data, size)
                .await
                .expect("ReLU failed"),
            1 => sigmoid(&dev.device, &dev.queue, &data, size)
                .await
                .expect("Sigmoid failed"),
            2 => tanh(&dev.device, &dev.queue, &data, size)
                .await
                .expect("Tanh failed"),
            _ => gelu(&dev.device, &dev.queue, &data, size)
                .await
                .expect("GELU failed"),
        };
    }

    assert_eq!(data.len(), size, "Should complete all operations");
}

#[tokio::test]
async fn test_deep_network_stack() {
    // Simulate deep network (100 layers)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 4;
    let dim = 256;
    let num_layers = 100;

    let mut x = vec![0.5f32; batch * dim];
    let ln_weights = vec![1.0f32; dim];
    let ln_bias = vec![0.0f32; dim];

    for _layer in 0..num_layers {
        // LayerNorm → ReLU (simplified layer)
        x = layer_norm(&dev.device, &dev.queue, &x, &ln_weights, &ln_bias, batch, dim, 1e-5)
            .await
            .expect("LayerNorm failed");

        x = relu(&dev.device, &dev.queue, &x, batch * dim)
            .await
            .expect("ReLU failed");
    }

    assert_eq!(x.len(), batch * dim, "Should complete deep network");
}

#[tokio::test]
async fn test_memory_intensive_concatenation() {
    // Concatenate many large tensors
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 16;
    let dim1 = 512;
    let dim2 = 512;

    let a = vec![0.5f32; batch * dim1];
    let b = vec![0.3f32; batch * dim2];

    // Repeat concatenation multiple times
    for _ in 0..10 {
        let result = concat(&dev.device, &dev.queue, &a, &b, batch, dim1, dim2).await;
        assert!(result.is_ok(), "Should handle large concatenation");
    }
}
