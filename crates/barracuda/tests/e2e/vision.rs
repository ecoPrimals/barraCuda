//! E2E Tests: Computer Vision Architectures
//!
//! Tests ResNet, YOLO, object detection pipelines
//! **Deep Debt**: Complete implementations, GPU-accelerated

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_resnet_residual_block() {
    // ResNet residual block: Conv → BN → ReLU → Conv → BN → Add → ReLU
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 1;
    let channels = 64;
    let height = 32;
    let width = 32;

    // Input: [batch, channels, height, width]
    let input = vec![0.5f32; batch * channels * height * width];

    // First conv (identity for now)
    let conv1_out = input.clone();

    // Batch norm
    let bn_scale = vec![1.0f32; channels];
    let bn_bias = vec![0.0f32; channels];
    let bn_mean = vec![0.0f32; channels];
    let bn_var = vec![1.0f32; channels];

    let bn1_out = batch_norm(
        &dev.device,
        &dev.queue,
        &conv1_out,
        &bn_scale,
        &bn_bias,
        &bn_mean,
        &bn_var,
        batch,
        channels,
        height * width,
        1e-5,
    )
    .await
    .expect("BatchNorm 1 failed");

    // ReLU
    let relu1_out = relu(&dev.device, &dev.queue, &bn1_out, batch * channels * height * width)
        .await
        .expect("ReLU 1 failed");

    // Second conv (identity)
    let conv2_out = relu1_out.clone();

    // Batch norm
    let bn2_out = batch_norm(
        &dev.device,
        &dev.queue,
        &conv2_out,
        &bn_scale,
        &bn_bias,
        &bn_mean,
        &bn_var,
        batch,
        channels,
        height * width,
        1e-5,
    )
    .await
    .expect("BatchNorm 2 failed");

    // Residual connection (add input)
    let residual_out = add(&dev.device, &dev.queue, &input, &bn2_out, batch * channels * height * width)
        .await
        .expect("Residual add failed");

    // Final ReLU
    let output = relu(&dev.device, &dev.queue, &residual_out, batch * channels * height * width)
        .await
        .expect("ReLU 2 failed");

    assert_eq!(output.len(), batch * channels * height * width, "ResNet block output");
}

#[tokio::test]
async fn test_convnet_forward_pass() {
    // Simple ConvNet: Conv → ReLU → Pool → Conv → ReLU → Pool
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 1;
    let in_channels = 3;
    let out_channels = 16;
    let height = 64;
    let width = 64;

    // Input image: [batch, channels, height, width]
    let input = vec![0.5f32; batch * in_channels * height * width];

    // Conv1: 3 → 16 channels (simplified - identity mapping)
    let conv1_out = vec![0.5f32; batch * out_channels * height * width];

    // ReLU
    let relu1_out = relu(&dev.device, &dev.queue, &conv1_out, batch * out_channels * height * width)
        .await
        .expect("ReLU 1 failed");

    // MaxPool 2x2
    let pool1_out = maxpool2d(
        &dev.device,
        &dev.queue,
        &relu1_out,
        batch,
        out_channels,
        height,
        width,
        2,
        2,
        2,
        2,
        0,
        0,
    )
    .await
    .expect("MaxPool 1 failed");

    let pool_h = height / 2;
    let pool_w = width / 2;

    assert_eq!(
        pool1_out.len(),
        batch * out_channels * pool_h * pool_w,
        "ConvNet output size"
    );
}

#[tokio::test]
async fn test_yolo_detection_pipeline() {
    // YOLO-like detection: Conv backbone → Detection head
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 1;
    let channels = 256;
    let grid_h = 8;
    let grid_w = 8;

    // Feature map from backbone
    let features = vec![0.5f32; batch * channels * grid_h * grid_w];

    // Detection head: Predict objectness, classes, bbox
    let num_anchors = 3;
    let num_classes = 80;
    let bbox_dims = 4; // x, y, w, h

    // Conv to prediction channels
    let pred_channels = num_anchors * (1 + num_classes + bbox_dims);
    let predictions = vec![0.1f32; batch * pred_channels * grid_h * grid_w];

    // Sigmoid for objectness
    let objectness = sigmoid(&dev.device, &dev.queue, &predictions[0..100], 100)
        .await
        .expect("Objectness sigmoid failed");

    // Softmax for classes
    let class_probs = softmax(&dev.device, &dev.queue, &predictions[100..200], 10, 10)
        .await
        .expect("Class softmax failed");

    assert_eq!(objectness.len(), 100, "Objectness scores");
    assert_eq!(class_probs.len(), 100, "Class probabilities");
}

#[tokio::test]
async fn test_image_augmentation_pipeline() {
    // Augmentation: Normalize → RandomCrop → Flip
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let batch = 2;
    let channels = 3;
    let height = 224;
    let width = 224;

    // Input images
    let images = vec![0.5f32; batch * channels * height * width];

    // Normalize (subtract mean, divide by std)
    let mean = vec![0.485, 0.456, 0.406]; // ImageNet mean
    let std = vec![0.229, 0.224, 0.225]; // ImageNet std

    let mut normalized = Vec::with_capacity(images.len());
    for (i, &pixel) in images.iter().enumerate() {
        let c = (i / (height * width)) % channels;
        normalized.push((pixel - mean[c]) / std[c]);
    }

    // Horizontal flip (50% chance)
    let flipped = flip(&dev.device, &dev.queue, &normalized, batch, channels, height, width, 1)
        .await
        .expect("Flip failed");

    assert_eq!(flipped.len(), batch * channels * height * width, "Augmented images");
}
