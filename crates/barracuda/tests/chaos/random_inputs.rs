//! Chaos Tests: Random Inputs
//!
//! Fuzz operations with random dimensions and data
//! **Deep Debt**: Operations should handle any valid input

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_matmul_random_dimensions() {
    // Test matmul with 100 random dimension combinations
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    for i in 0..100 {
        // Random dimensions (1..256)
        let m = 1 + (i * 7) % 256;
        let n = 1 + (i * 11) % 256;
        let k = 1 + (i * 13) % 256;

        let a = vec![0.5f32; m * k];
        let b = vec![0.3f32; k * n];

        let result = matmul(&dev.device, &dev.queue, &a, &b, m, k, n).await;

        assert!(
            result.is_ok(),
            "Matmul should handle dimensions m={}, n={}, k={}",
            m,
            n,
            k
        );

        if let Ok(output) = result {
            assert_eq!(
                output.len(),
                m * n,
                "Matmul output size should be m * n = {} * {} = {}",
                m,
                n,
                m * n
            );
        }
    }
}

#[tokio::test]
async fn test_relu_random_sizes() {
    // Test ReLU with random input sizes
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    for i in 0..50 {
        let size = 1 + (i * 17) % 10000;
        let input = vec![-0.5f32; size];

        let result = relu(&dev.device, &dev.queue, &input, size).await;

        assert!(
            result.is_ok(),
            "ReLU should handle size={}",
            size
        );

        if let Ok(output) = result {
            assert_eq!(output.len(), size);
            // All outputs should be non-negative (ReLU property)
            assert!(
                output.iter().all(|&x| x >= 0.0),
                "ReLU should output non-negative values"
            );
        }
    }
}

#[tokio::test]
async fn test_softmax_random_shapes() {
    // Test softmax with random batch/class dimensions
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    for i in 0..50 {
        let batch = 1 + (i * 3) % 64;
        let classes = 2 + (i * 5) % 128;
        let size = batch * classes;

        let input = vec![0.5f32; size];

        let result = softmax(&dev.device, &dev.queue, &input, batch, classes).await;

        assert!(
            result.is_ok(),
            "Softmax should handle batch={}, classes={}",
            batch,
            classes
        );

        if let Ok(output) = result {
            assert_eq!(output.len(), size);
            
            // Check probabilities sum to ~1.0 per batch
            for b in 0..batch {
                let batch_sum: f32 = output[b * classes..(b + 1) * classes].iter().sum();
                assert!(
                    (batch_sum - 1.0).abs() < 0.01,
                    "Softmax probabilities should sum to 1.0, got {}",
                    batch_sum
                );
            }
        }
    }
}

#[tokio::test]
async fn test_conv2d_random_parameters() {
    // Test Conv2D with random kernel/stride/padding
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    for i in 0..20 {
        let batch = 1 + i % 4;
        let in_channels = 1 + i % 16;
        let out_channels = 1 + (i * 2) % 32;
        let height = 8 + i * 4;
        let width = 8 + i * 4;
        let kernel_size = 1 + i % 5;
        let stride = 1 + i % 3;
        let padding = i % 2;

        let input_size = batch * in_channels * height * width;
        let kernel_size_total = out_channels * in_channels * kernel_size * kernel_size;

        let input = vec![0.5f32; input_size];
        let weights = vec![0.1f32; kernel_size_total];
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
            kernel_size,
            kernel_size,
            stride,
            stride,
            padding,
            padding,
        )
        .await;

        assert!(
            result.is_ok(),
            "Conv2D should handle batch={}, in_c={}, out_c={}, h={}, w={}, k={}, s={}, p={}",
            batch,
            in_channels,
            out_channels,
            height,
            width,
            kernel_size,
            stride,
            padding
        );
    }
}

#[tokio::test]
async fn test_add_random_vectors() {
    // Test element-wise add with random sizes
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    for i in 0..100 {
        let size = 1 + (i * 23) % 1000;
        let a = vec![0.5f32; size];
        let b = vec![0.3f32; size];

        let result = add(&dev.device, &dev.queue, &a, &b, size).await;

        assert!(result.is_ok(), "Add should handle size={}", size);

        if let Ok(output) = result {
            assert_eq!(output.len(), size);
            // Check correctness: a[i] + b[i] = 0.5 + 0.3 = 0.8
            assert!(
                (output[0] - 0.8).abs() < 0.001,
                "Add should compute correct sum"
            );
        }
    }
}
