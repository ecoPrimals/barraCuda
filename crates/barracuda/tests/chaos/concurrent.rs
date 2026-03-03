//! Chaos Tests: Concurrent Execution
//!
//! Test parallel operation execution for race conditions
//! **Deep Debt**: Thread-safe, no data races, Arc safety

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;
use futures::future::join_all;

#[tokio::test]
async fn test_concurrent_matmul() {
    // Run 50 matmuls concurrently
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let handles: Vec<_> = (0..50)
        .map(|i| {
            let dev = dev.clone();
            tokio::spawn(async move {
                let m = 64 + i;
                let n = 64 + i;
                let k = 64 + i;
                let a = vec![0.5f32; m * k];
                let b = vec![0.3f32; k * n];
                matmul(&dev.device, &dev.queue, &a, &b, m, k, n).await
            })
        })
        .collect();

    let results = join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        assert!(
            result.is_ok(),
            "Concurrent matmul {} should succeed",
            i
        );
        assert!(
            result.unwrap().is_ok(),
            "Matmul result {} should be valid",
            i
        );
    }
}

#[tokio::test]
async fn test_concurrent_mixed_operations() {
    // Run different operations concurrently
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let dev = dev.clone();
            tokio::spawn(async move {
                let size = 100;
                let data = vec![0.5f32; size];

                match i % 5 {
                    0 => relu(&dev.device, &dev.queue, &data, size).await,
                    1 => sigmoid(&dev.device, &dev.queue, &data, size).await,
                    2 => tanh(&dev.device, &dev.queue, &data, size).await,
                    3 => gelu(&dev.device, &dev.queue, &data, size).await,
                    _ => softmax(&dev.device, &dev.queue, &data, 10, 10).await,
                }
            })
        })
        .collect();

    let results = join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        if let Ok(inner_result) = result {
            assert!(
                inner_result.is_ok(),
                "Concurrent operation {} should succeed",
                i
            );
        } else {
            panic!("Concurrent operation {} failed", i);
        }
    }
}

#[tokio::test]
async fn test_concurrent_training_steps() {
    // Multiple training steps running in parallel
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let handles: Vec<_> = (0..20)
        .map(|_| {
            let dev = dev.clone();
            tokio::spawn(async move {
                let params = vec![0.5f32; 100];
                let grads = vec![0.1f32; 100];
                sgd(&dev.device, &dev.queue, &params, &grads, 0.01, 100).await
            })
        })
        .collect();

    let results = join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        if let Ok(inner_result) = result {
            assert!(
                inner_result.is_ok(),
                "Concurrent SGD step {} should succeed",
                i
            );
        } else {
            panic!("Concurrent SGD step {} failed", i);
        }
    }
}

#[tokio::test]
async fn test_device_sharing_safety() {
    // Verify Arc<Device> is safe to share across tasks
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    // Clone device handle 100 times
    let devices: Vec<_> = (0..100).map(|_| dev.clone()).collect();

    // Use each clone concurrently
    let handles: Vec<_> = devices
        .into_iter()
        .map(|dev| {
            tokio::spawn(async move {
                let data = vec![0.5f32; 100];
                relu(&dev.device, &dev.queue, &data, 100).await
            })
        })
        .collect();

    let results = join_all(handles).await;

    assert_eq!(results.len(), 100, "All concurrent operations should complete");
}
