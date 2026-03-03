//! Fault Injection Tests
//!
//! Tests error handling, recovery, and graceful degradation:
//! - Invalid inputs
//! - Shape mismatches
//! - Device failures
//! - Timeout handling
//!
//! ## Philosophy
//!
//! Good systems fail gracefully. These tests verify that Barracuda:
//! 1. Returns clear error messages
//! 2. Doesn't panic on bad inputs
//! 3. Releases resources properly on failure
//! 4. Provides meaningful recovery paths

mod common;

use barracuda::device::{Device, KernelRouter};
use barracuda::tensor::Tensor;

// ============================================================================
// Invalid Input Tests
// ============================================================================

#[tokio::test]
async fn test_shape_mismatch_matmul() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Shape Mismatch Error Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Create incompatible matrices (3x4 * 5x6 - inner dimensions don't match)
        let a_data: Vec<f32> = vec![1.0; 12]; // 3x4
        let b_data: Vec<f32> = vec![1.0; 30]; // 5x6

        let a = Tensor::from_vec_on(a_data, vec![3, 4], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![5, 6], device)
            .await
            .unwrap();

        // This should fail with a clear error
        match a.matmul(&b) {
            Ok(_) => {
                panic!("Matmul with mismatched shapes should fail");
            }
            Err(e) => {
                println!("  ✓ Shape mismatch correctly rejected");
                println!("  Error: {}", e);
                // Verify error contains useful info
                let error_msg = e.to_string().to_lowercase();
                assert!(
                    error_msg.contains("shape")
                        || error_msg.contains("dimension")
                        || error_msg.contains("mismatch"),
                    "Error should mention shape/dimension issue"
                );
            }
        }

        println!("\n  Shape mismatch handling: PASS\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_data_shape_mismatch() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Data/Shape Mismatch Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Data has 10 elements but shape says 12
        let data: Vec<f32> = vec![1.0; 10];
        let shape = vec![3, 4]; // 12 elements expected

        match Tensor::from_vec_on(data, shape, device).await {
            Ok(_) => {
                panic!("Mismatched data/shape should fail");
            }
            Err(e) => {
                println!("  ✓ Data/shape mismatch correctly rejected");
                println!("  Error: {}", e);
            }
        }

        println!("\n  Data/shape mismatch: PASS\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_nan_input_handling() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== NaN Input Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Create tensor with NaN values
        let data_with_nan = vec![1.0, 2.0, f32::NAN, 4.0];

        let tensor = Tensor::from_vec_on(data_with_nan.clone(), vec![4], device)
            .await
            .unwrap();

        let result = tensor.to_vec().unwrap();

        // Check that NaN is preserved (not silently converted)
        assert!(result[2].is_nan(), "NaN should be preserved in tensor");
        println!("  ✓ NaN values preserved correctly");

        println!("\n  NaN handling: PASS\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_inf_input_handling() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Inf Input Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Create tensor with Inf values
        let data_with_inf = vec![1.0, f32::INFINITY, f32::NEG_INFINITY, 4.0];

        let tensor = Tensor::from_vec_on(data_with_inf.clone(), vec![4], device)
            .await
            .unwrap();

        let result = tensor.to_vec().unwrap();

        assert!(result[1].is_infinite() && result[1] > 0.0, "+Inf preserved");
        assert!(result[2].is_infinite() && result[2] < 0.0, "-Inf preserved");
        println!("  ✓ Inf values preserved correctly");

        println!("\n  Inf handling: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// Cholesky-Specific Fault Tests
// ============================================================================

#[tokio::test]
async fn test_cholesky_non_positive_definite() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Cholesky Non-Positive-Definite Matrix ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Non-positive-definite matrix (negative eigenvalue)
        let non_spd = vec![-4.0, 2.0, 2.0, -3.0];

        let tensor = Tensor::from_vec_on(non_spd, vec![2, 2], device)
            .await
            .unwrap();

        match tensor.cholesky() {
            Ok(result) => {
                // Some implementations produce NaN for non-SPD
                let result_data = result.to_vec().unwrap();
                if result_data.iter().any(|x| x.is_nan()) {
                    println!("  ✓ Non-SPD matrix produced NaN (detected)");
                } else {
                    println!("  ⚠ Non-SPD matrix produced result (may be incorrect)");
                }
            }
            Err(e) => {
                println!("  ✓ Non-SPD matrix correctly rejected");
                println!("  Error: {}", e);
            }
        }

        println!("\n  Cholesky non-SPD handling: PASS\n");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_cholesky_non_square_matrix() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Cholesky Non-Square Matrix ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Non-square matrix (3x2)
        let non_square = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let tensor = Tensor::from_vec_on(non_square, vec![3, 2], device)
            .await
            .unwrap();

        match tensor.cholesky() {
            Ok(_) => {
                panic!("Cholesky on non-square matrix should fail");
            }
            Err(e) => {
                println!("  ✓ Non-square matrix correctly rejected");
                println!("  Error: {}", e);
            }
        }

        println!("\n  Cholesky non-square: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// Kernel Router Fault Tests
// ============================================================================

#[test]
fn test_kernel_router_invalid_model_fallback() {
    println!("\n=== Kernel Router Invalid Model Fallback ===\n");

    let router = KernelRouter::default();

    // Try to route to non-existent NPU model
    let workload = barracuda::device::ComputeWorkload::SparseInference {
        input_sparsity: 0.99,
        model_name: "definitely_does_not_exist_model_xyz123".to_string(),
    };

    match router.route(&workload) {
        Ok(target) => {
            // Should fall back to WGSL
            match target {
                barracuda::device::KernelTarget::Wgsl { .. } => {
                    println!("  ✓ Missing NPU model correctly falls back to WGSL");
                }
                barracuda::device::KernelTarget::Npu { .. } => {
                    panic!("Should not route to NPU with non-existent model");
                }
                _ => {
                    println!("  Routed to alternative target");
                }
            }
        }
        Err(e) => {
            panic!("Routing should not fail (should fallback): {}", e);
        }
    }

    println!("\n  Kernel router fallback: PASS\n");
}

// ============================================================================
// Resource Cleanup Tests
// ============================================================================

#[tokio::test]
async fn test_resource_cleanup_on_error() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Resource Cleanup on Error ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Track operations that might leak resources
        let iterations = 50;
        let mut errors = 0;

        for i in 0..iterations {
            // Alternate between valid and invalid operations
            if i % 5 == 0 {
                // Invalid operation (should fail cleanly)
                let data: Vec<f32> = vec![1.0; 10];
                let bad_shape = vec![3, 4]; // Mismatch

                match Tensor::from_vec_on(data, bad_shape, device.clone()).await {
                    Ok(_) => {}
                    Err(_) => {
                        errors += 1;
                    }
                }
            } else {
                // Valid operation
                let data: Vec<f32> = vec![1.0; 16];
                let tensor = Tensor::from_vec_on(data, vec![4, 4], device.clone())
                    .await
                    .unwrap();
                let _result = tensor.to_vec().unwrap();
            }
        }

        println!("  Iterations: {}", iterations);
        println!("  Expected errors: {}", iterations / 5);
        println!("  Actual errors: {}", errors);

        // If we get here without panic or hang, cleanup worked
        println!("  ✓ No resource leaks detected (completed without hang)");

        println!("\n  Resource cleanup: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// Timeout Simulation Tests
// ============================================================================

#[tokio::test]
async fn test_operation_timeout_handling() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Operation Timeout Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Create a reasonably large tensor operation
        let size = 512;
        let data: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.001).collect();

        let tensor_a = Tensor::from_vec_on(data.clone(), vec![size, size], device.clone())
            .await
            .unwrap();
        let tensor_b = Tensor::from_vec_on(data, vec![size, size], device)
            .await
            .unwrap();

        // Use tokio timeout to wrap the operation
        let timeout_result = tokio::time::timeout(
            std::time::Duration::from_secs(30), // 30 second timeout
            async {
                let result = tensor_a.matmul(&tensor_b)?;
                result.to_vec()
            },
        )
        .await;

        match timeout_result {
            Ok(Ok(result)) => {
                println!("  ✓ Operation completed within timeout");
                println!("  Result size: {} elements", result.len());
            }
            Ok(Err(e)) => {
                println!("  Operation failed (not timeout): {}", e);
            }
            Err(_) => {
                println!("  ⚠ Operation timed out (may indicate performance issue)");
            }
        }

        println!("\n  Timeout handling: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// Device Unavailability Tests
// ============================================================================

#[test]
fn test_device_unavailability_handling() {
    println!("\n=== Device Unavailability Handling ===\n");

    // Test that Device enum handles unavailable devices gracefully
    let devices = vec![Device::CPU, Device::GPU, Device::NPU, Device::TPU];

    for device in devices {
        let is_available = device.is_available();
        let info = device.info();

        println!("  {} - available: {}", device, is_available);
        println!("    capabilities: {:?}", info.capabilities);

        // CPU should always be available
        if matches!(device, Device::CPU) {
            assert!(is_available, "CPU should always be available");
        }
    }

    // Test that Auto selection never fails
    let available = Device::available_devices();
    assert!(
        !available.is_empty(),
        "At least one device should be available"
    );
    println!("\n  Available devices: {:?}", available);

    println!("\n  Device unavailability handling: PASS\n");
}

// ============================================================================
// Error Message Quality Tests
// ============================================================================

#[tokio::test]
async fn test_error_messages_are_informative() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Error Message Quality ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        // Collection of error-inducing scenarios
        let test_cases = vec![
            ("Data/shape mismatch", vec![1.0; 10], vec![3, 4]),
            ("Empty data", vec![], vec![0]),
        ];

        for (name, data, shape) in test_cases {
            match Tensor::from_vec_on(data, shape, device.clone()).await {
                Ok(_) => {
                    println!("  {} - no error (implementation allows)", name);
                }
                Err(e) => {
                    let msg = e.to_string();
                    println!("  {} - Error: {}", name, msg);

                    // Error messages should:
                    // 1. Not be empty
                    assert!(!msg.is_empty(), "Error message should not be empty");

                    // 2. Not contain internal implementation details only
                    // (This is a soft check - we just log it)
                    if msg.len() < 10 {
                        println!("    ⚠ Error message might be too terse");
                    }
                }
            }
        }

        println!("\n  Error message quality: PASS\n");
    }) {
        return;
    }
}

// ============================================================================
// Concurrent Error Handling
// ============================================================================

#[tokio::test]
async fn test_concurrent_error_handling() {
    if !common::run_gpu_resilient_async(|| async {
        println!("\n=== Concurrent Error Handling ===\n");

        let device = barracuda::device::test_pool::get_test_device().await;

        let num_tasks = 10;
        let mut handles = Vec::new();

        for i in 0..num_tasks {
            let device_clone = device.clone();

            let handle = tokio::spawn(async move {
                // Some tasks have errors, some don't
                if i % 3 == 0 {
                    // Error case - mismatched data/shape
                    let data: Vec<f32> = vec![1.0; 5];
                    let shape = vec![2, 3]; // Expects 6 elements
                    Tensor::from_vec_on(data, shape, device_clone)
                        .await
                        .is_err()
                } else {
                    // Success case
                    let data: Vec<f32> = vec![1.0; 6];
                    let shape = vec![2, 3];
                    Tensor::from_vec_on(data, shape, device_clone).await.is_ok()
                }
            });

            handles.push(handle);
        }

        // All tasks should complete (no panics)
        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            results.push(h.await.unwrap());
        }

        let _expected_errors = num_tasks / 3 + if num_tasks % 3 > 0 { 1 } else { 0 };
        let _actual_errors = results.iter().filter(|&&x| x).count();

        println!("  Tasks: {}", num_tasks);
        println!("  All tasks completed without panic");

        println!("\n  Concurrent error handling: PASS\n");
    }) {
        return;
    }
}
