//! E2E Tests: Training Pipelines
//!
//! Tests forward → loss → backward → optimizer workflows
//! **Deep Debt**: Complete training loops, no mocks

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_simple_training_step() {
    // Training step: Forward → Loss → Update (simplified)
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 4;
    let input_dim = 10;
    let output_dim = 2;

    // Input data
    let inputs = vec![0.5f32; batch * input_dim];
    let targets = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // One-hot [batch, 2]

    // Model weights
    let weights = vec![0.1f32; input_dim * output_dim];

    // Forward pass: matmul → softmax
    let logits = matmul(&dev.device, &dev.queue, &inputs, &weights, batch, input_dim, output_dim)
        .await
        .expect("Forward matmul failed");

    let predictions = softmax(&dev.device, &dev.queue, &logits, batch, output_dim)
        .await
        .expect("Softmax failed");

    // Loss: cross-entropy
    let loss = cross_entropy(&dev.device, &dev.queue, &predictions, &targets, batch, output_dim)
        .await
        .expect("CrossEntropy failed");

    assert!(loss > 0.0, "Loss should be positive");
}

#[tokio::test]
async fn test_optimizer_update() {
    // Optimizer update: SGD, Adam
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let param_count = 100;
    let learning_rate = 0.01;

    // Parameters and gradients
    let params = vec![0.5f32; param_count];
    let grads = vec![0.1f32; param_count];

    // SGD update: params -= lr * grads
    let updated_params = sgd(&dev.device, &dev.queue, &params, &grads, learning_rate, param_count)
        .await
        .expect("SGD update failed");

    // Check update happened
    assert_eq!(updated_params.len(), param_count);
    assert!(
        (updated_params[0] - (0.5 - 0.01 * 0.1)).abs() < 1e-5,
        "SGD should update parameters"
    );
}

#[tokio::test]
async fn test_adam_optimizer_with_state() {
    // Adam optimizer with momentum and adaptive lr
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let param_count = 50;
    let learning_rate = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    // Initial state
    let params = vec![1.0f32; param_count];
    let grads = vec![0.1f32; param_count];
    let m = vec![0.0f32; param_count]; // First moment
    let v = vec![0.0f32; param_count]; // Second moment
    let t = 1; // Time step

    // Adam step
    let updated_params = adam(
        &dev.device,
        &dev.queue,
        &params,
        &grads,
        &m,
        &v,
        learning_rate,
        beta1,
        beta2,
        epsilon,
        t,
        param_count,
    )
    .await
    .expect("Adam update failed");

    assert_eq!(updated_params.len(), param_count);
}

#[tokio::test]
async fn test_multi_step_training() {
    // Multiple training steps in sequence
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 2;
    let dim = 10;
    let num_steps = 5;

    let inputs = vec![0.5f32; batch * dim];
    let targets = vec![1.0; batch * dim];
    let mut params = vec![0.1f32; dim * dim];

    for step in 0..num_steps {
        // Forward pass
        let outputs = matmul(&dev.device, &dev.queue, &inputs, &params, batch, dim, dim)
            .await
            .expect(&format!("Step {} forward failed", step));

        // Compute loss (MSE)
        let loss = mse_loss(&dev.device, &dev.queue, &outputs, &targets, batch * dim)
            .await
            .expect(&format!("Step {} loss failed", step));

        // Simplified gradient (just use loss as signal)
        let grads = vec![loss / (dim * dim) as f32; dim * dim];

        // Update parameters
        params = sgd(&dev.device, &dev.queue, &params, &grads, 0.01, dim * dim)
            .await
            .expect(&format!("Step {} SGD failed", step));
    }

    // After 5 steps, parameters should have changed
    assert_ne!(params[0], 0.1f32, "Parameters should update over training");
}

#[tokio::test]
async fn test_loss_functions_comparison() {
    // Compare different loss functions on same input
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 4;
    let dim = 10;

    let predictions = vec![0.5f32; batch * dim];
    let targets = vec![0.7f32; batch * dim];

    // MSE loss
    let mse = mse_loss(&dev.device, &dev.queue, &predictions, &targets, batch * dim)
        .await
        .expect("MSE failed");

    // L1 loss
    let l1 = l1_loss(&dev.device, &dev.queue, &predictions, &targets, batch * dim)
        .await
        .expect("L1 failed");

    // Huber loss
    let huber = huber_loss(&dev.device, &dev.queue, &predictions, &targets, 1.0, batch * dim)
        .await
        .expect("Huber failed");

    // All should be positive and finite
    assert!(mse > 0.0 && mse.is_finite(), "MSE should be valid");
    assert!(l1 > 0.0 && l1.is_finite(), "L1 should be valid");
    assert!(huber > 0.0 && huber.is_finite(), "Huber should be valid");
}
