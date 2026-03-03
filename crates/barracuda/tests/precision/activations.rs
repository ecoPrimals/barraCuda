// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision Tests: Activation Functions
//!
//! Validate FP32 precision for GELU, Sigmoid, Tanh, etc.
//! **Deep Debt**: Transcendental functions within FP32 tolerance

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

/// CPU reference for sigmoid: 1 / (1 + exp(-x))
fn cpu_sigmoid_reference(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

#[tokio::test]
async fn test_sigmoid_precision() {
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();

    let gpu_result = sigmoid(&dev.device, &dev.queue, &input, size)
        .await
        .expect("GPU sigmoid failed");

    let cpu_result = cpu_sigmoid_reference(&input);

    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "Sigmoid max error should be < 1e-5, got {}",
        max_error
    );
}

/// CPU reference for tanh
fn cpu_tanh_reference(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.tanh()).collect()
}

#[tokio::test]
async fn test_tanh_precision() {
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();

    let gpu_result = tanh(&dev.device, &dev.queue, &input, size)
        .await
        .expect("GPU tanh failed");

    let cpu_result = cpu_tanh_reference(&input);

    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "Tanh max error should be < 1e-5, got {}",
        max_error
    );
}

/// CPU reference for GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
fn cpu_gelu_reference(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let inner = sqrt_2_over_pi * (v + 0.044715 * v.powi(3));
            v * 0.5 * (1.0 + inner.tanh())
        })
        .collect()
}

#[tokio::test]
async fn test_gelu_precision() {
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let size = 1000;
    let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 - 5.0).collect();

    let gpu_result = gelu(&dev.device, &dev.queue, &input, size)
        .await
        .expect("GPU GELU failed");

    let cpu_result = cpu_gelu_reference(&input);

    let max_error = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-4,
        "GELU max error should be < 1e-4, got {}",
        max_error
    );
}
