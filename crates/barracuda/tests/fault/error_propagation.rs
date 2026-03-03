//! Fault Tests: Error Propagation
//!
//! Test that errors propagate correctly through pipelines
//! **Deep Debt**: Result<> composition, no silent failures

use barracuda::device::test_pool::get_test_device;
use barracuda::ops::*;

#[tokio::test]
async fn test_pipeline_error_stops_execution() {
    // If one operation fails, pipeline should stop
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let input = vec![0.5f32; 100];

    // Step 1: Valid ReLU
    let step1 = relu(&dev.device, &dev.queue, &input, 100).await;
    assert!(step1.is_ok(), "Step 1 should succeed");

    // Step 2: Invalid softmax (zero classes) - should error
    let step2 = step1.and_then(|data| async {
        softmax(&dev.device, &dev.queue, &data, 100, 0).await
    }.await);

    assert!(step2.is_err(), "Pipeline should fail at invalid step");
}

#[tokio::test]
async fn test_error_contains_context() {
    // Errors should have meaningful messages
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    let a = vec![0.5f32; 10];
    let b = vec![0.3f32; 20]; // Mismatched!

    let result = matmul(&dev.device, &dev.queue, &a, &b, 10, 10, 20).await;

    if let Err(e) = result {
        let error_msg = format!("{}", e);
        // Error should mention "dimension" or "shape" or "size"
        assert!(
            error_msg.to_lowercase().contains("dimension")
                || error_msg.to_lowercase().contains("shape")
                || error_msg.to_lowercase().contains("size"),
            "Error message should be descriptive: {}",
            error_msg
        );
    }
}

#[tokio::test]
async fn test_recoverable_error_allows_retry() {
    // After an error, device should still work for next operation
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    // First operation: intentionally fail
    let _fail = softmax(&dev.device, &dev.queue, &vec![0.5; 10], 10, 0).await;

    // Second operation: should still work
    let success = relu(&dev.device, &dev.queue, &vec![0.5; 10], 10).await;

    assert!(
        success.is_ok(),
        "Device should recover after error"
    );
}

#[tokio::test]
async fn test_multiple_errors_independent() {
    // Multiple errors should not affect each other
    let Some(dev) = get_test_device_if_gpu_available().await else { return };
    // Error 1
    let err1 = softmax(&dev.device, &dev.queue, &vec![], 0, 0).await;

    // Error 2
    let err2 = matmul(&dev.device, &dev.queue, &vec![], &vec![], 0, 0, 0).await;

    // Both should error independently
    assert!(err1.is_err(), "Error 1 should occur");
    assert!(err2.is_err(), "Error 2 should occur");

    // Valid operation should still work
    let ok = relu(&dev.device, &dev.queue, &vec![0.5; 10], 10).await;
    assert!(ok.is_ok(), "Valid operation should work after errors");
}
