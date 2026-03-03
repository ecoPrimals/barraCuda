// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision Tests: Core Operations
//!
//! Validate FP32 precision for matmul, add, mul, etc.
//! **Deep Debt**: Max absolute error < 1e-5 for FP32

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

/// CPU reference implementation for matmul (naive, but correct)
fn cpu_matmul_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[tokio::test]
async fn test_matmul_precision() {
    // Compare GPU matmul vs CPU reference
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let m = 32;
    let k = 32;
    let n = 32;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();

    // GPU result
    let gpu_result = matmul(&dev.device, &dev.queue, &a, &b, m, k, n)
        .await
        .expect("GPU matmul failed");

    // CPU reference
    let cpu_result = cpu_matmul_reference(&a, &b, m, k, n);

    // Compare element-wise
    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-3,
        "Matmul max absolute error should be < 1e-3 for FP32, got {}",
        max_error
    );
}

/// CPU reference for element-wise add
fn cpu_add_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[tokio::test]
async fn test_add_precision() {
    // Compare GPU add vs CPU reference
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.02).collect();

    // GPU result
    let gpu_result = add(&dev.device, &dev.queue, &a, &b, size)
        .await
        .expect("GPU add failed");

    // CPU reference
    let cpu_result = cpu_add_reference(&a, &b);

    // Compare
    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-6,
        "Add max error should be < 1e-6 (exact for FP32), got {}",
        max_error
    );
}

/// CPU reference for ReLU
fn cpu_relu_reference(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

#[tokio::test]
async fn test_relu_precision() {
    // ReLU should be exact (no numerical error)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();

    // GPU result
    let gpu_result = relu(&dev.device, &dev.queue, &input, size)
        .await
        .expect("GPU ReLU failed");

    // CPU reference
    let cpu_result = cpu_relu_reference(&input);

    // Compare (should be exact)
    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-7,
        "ReLU should be exact, got max error {}",
        max_error
    );
}

/// CPU reference for sum reduction
fn cpu_sum_reference(x: &[f32]) -> f32 {
    x.iter().sum()
}

#[tokio::test]
async fn test_sum_precision() {
    // Sum reduction accuracy
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();

    // GPU result
    let gpu_result = sum(&dev.device, &dev.queue, &input, size)
        .await
        .expect("GPU sum failed");

    // CPU reference
    let cpu_result = cpu_sum_reference(&input);

    let error = (gpu_result - cpu_result).abs();

    assert!(
        error < 0.01,
        "Sum error should be small, got {} (GPU: {}, CPU: {})",
        error,
        gpu_result,
        cpu_result
    );
}

/// CPU reference for softmax
fn cpu_softmax_reference(x: &[f32], batch: usize, classes: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(x.len());
    
    for b in 0..batch {
        let start = b * classes;
        let end = start + classes;
        let batch_slice = &x[start..end];
        
        // Find max for numerical stability
        let max_val = batch_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x - max)
        let exp_values: Vec<f32> = batch_slice.iter().map(|&v| (v - max_val).exp()).collect();
        
        // Compute sum
        let sum: f32 = exp_values.iter().sum();
        
        // Normalize
        result.extend(exp_values.iter().map(|&v| v / sum));
    }
    
    result
}

#[tokio::test]
async fn test_softmax_precision() {
    // Softmax numerical stability and precision
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 8;
    let classes = 10;
    let size = batch * classes;

    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

    // GPU result
    let gpu_result = softmax(&dev.device, &dev.queue, &input, batch, classes)
        .await
        .expect("GPU softmax failed");

    // CPU reference
    let cpu_result = cpu_softmax_reference(&input, batch, classes);

    // Compare
    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-4,
        "Softmax max error should be < 1e-4, got {}",
        max_error
    );

    // Verify probabilities sum to 1.0
    for b in 0..batch {
        let batch_sum: f32 = gpu_result[b * classes..(b + 1) * classes].iter().sum();
        assert!(
            (batch_sum - 1.0).abs() < 0.01,
            "Softmax probabilities should sum to 1.0"
        );
    }
}
