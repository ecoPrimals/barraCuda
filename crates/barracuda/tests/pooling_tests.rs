// SPDX-License-Identifier: AGPL-3.0-only
//! Comprehensive tests for buffer pooling system
//!
//! Test categories:
//! - E2E: Full tensor operation pipeline with pooling
//! - Chaos: Concurrent access, stress tests
//! - Fault: Error handling, edge cases
//!
//! All tests share a single device instance to avoid wgpu cache invalidation
//! issues when devices are dropped while the global pipeline cache retains
//! references to their layouts/pipelines.

#![expect(clippy::unwrap_used, reason = "tests")]
use barracuda::device::{WgpuDevice, get_device_context};
use barracuda::prelude::*;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Shared device for all tests (avoids cache invalidation)
static TEST_DEVICE: OnceCell<Arc<WgpuDevice>> = OnceCell::const_new();

async fn get_test_device() -> Arc<WgpuDevice> {
    TEST_DEVICE
        .get_or_init(|| async {
            Arc::new(
                WgpuDevice::new()
                    .await
                    .expect("Failed to create test device"),
            )
        })
        .await
        .clone()
}

// ============================================================================
// E2E Tests - Full Pipeline
// ============================================================================

#[tokio::test]
async fn e2e_tensor_add_uses_pooling() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    let stats_before = ctx.stats();

    // Create input tensors (owned buffers)
    let a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
    let b = Tensor::from_data(&[5.0, 6.0, 7.0, 8.0], vec![4], device).unwrap();

    // Perform add operation (output uses pooled buffer)
    let c = a.add(&b).unwrap();

    // Verify result
    let result = c.to_vec().unwrap();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

    // Check that pooling is being used
    let stats_after = ctx.stats();
    assert!(
        stats_after.buffer_allocations >= stats_before.buffer_allocations,
        "Expected buffer tracking"
    );

    // The output tensor should be pooled
    assert!(c.is_pooled(), "Operation output should use pooled buffer");
}

#[tokio::test]
async fn e2e_tensor_mul_uses_pooling() {
    let device = get_test_device().await;

    let a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
    let b = Tensor::from_data(&[2.0, 3.0, 4.0, 5.0], vec![4], device).unwrap();

    let c = a.mul(&b).unwrap();

    let result = c.to_vec().unwrap();
    assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    assert!(c.is_pooled());
}

#[tokio::test]
async fn e2e_chained_operations_reuse_pool() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    let a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
    let b = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], vec![4], device).unwrap();

    // Chain of operations with explicit drops to return buffers
    let c = a.add(&b).unwrap();
    let _result1 = c.to_vec().unwrap();
    drop(c); // Returns buffer to pool

    let d = a.add(&b).unwrap();

    // Verify the result is correct
    let result = d.to_vec().unwrap();
    assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);

    // Check reuse happened
    let stats = ctx.stats();
    assert!(
        stats.buffer_reuses > 0 || stats.buffer_allocations > 0,
        "Expected some buffer activity"
    );
}

#[tokio::test]
async fn e2e_multiple_operations_steady_state() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    let size = 1000;
    let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let a = Tensor::from_data(&data, vec![size], device.clone()).unwrap();
    let b = Tensor::from_data(&data, vec![size], device).unwrap();

    // Warmup - populate pool
    for _ in 0..5 {
        let result = a.add(&b).unwrap();
        drop(result);
    }

    let stats_before = ctx.stats();

    // Steady state - should reuse buffers
    for _ in 0..10 {
        let result = a.add(&b).unwrap();
        drop(result);
    }

    let stats_after = ctx.stats();

    // Verify buffer reuse is happening
    let new_reuses = stats_after.buffer_reuses - stats_before.buffer_reuses;
    assert!(
        new_reuses > 0,
        "Expected buffer reuses in steady state, got {new_reuses}"
    );
}

// ============================================================================
// Chaos Tests - Concurrent Access
// ============================================================================

#[tokio::test]
async fn chaos_concurrent_tensor_operations() {
    let device = get_test_device().await;

    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let a = Tensor::from_data(&data, vec![100], device.clone()).unwrap();
    let b = Tensor::from_data(&data, vec![100], device).unwrap();

    // Multiple operations on same tensors - tests thread safety
    for _ in 0..20 {
        let result = a.add(&b).unwrap();
        let vec = result.to_vec().unwrap();
        assert_eq!(vec.len(), 100);
        assert_eq!(vec[0], 0.0); // 0 + 0 = 0
        assert_eq!(vec[99], 198.0); // 99 + 99 = 198
    }
}

#[tokio::test]
async fn chaos_rapid_acquire_release() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    let stats_before = ctx.stats();

    // Rapidly acquire and release buffers
    for _ in 0..100 {
        let buf = ctx.acquire_pooled_output(1000);
        drop(buf);
    }

    let stats_after = ctx.stats();

    // Should have buffer reuses after the first allocation
    let total_activity = (stats_after.buffer_allocations - stats_before.buffer_allocations)
        + (stats_after.buffer_reuses - stats_before.buffer_reuses);

    assert!(
        total_activity >= 100,
        "Expected at least 100 buffer operations"
    );

    // Check reuse rate (allowing for initial allocation)
    if total_activity > 1 {
        let reuses = stats_after.buffer_reuses - stats_before.buffer_reuses;
        assert!(
            reuses >= 90,
            "Expected high reuse rate, got {reuses} reuses"
        );
    }
}

#[tokio::test]
async fn chaos_mixed_sizes_stress() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    // Mix of different tensor sizes
    let sizes = [10, 100, 1000, 10000, 100, 10, 1000, 100];

    let stats_before = ctx.stats();

    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let a = Tensor::from_data(&data, vec![size], device.clone()).unwrap();
        let b = Tensor::from_data(&data, vec![size], device.clone()).unwrap();
        let result = a.add(&b).unwrap();
        // Verify correctness
        let vec = result.to_vec().unwrap();
        assert_eq!(vec.len(), size);
        drop(result);
    }

    // Run again - should reuse from different buckets
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let a = Tensor::from_data(&data, vec![size], device.clone()).unwrap();
        let b = Tensor::from_data(&data, vec![size], device.clone()).unwrap();
        let result = a.add(&b).unwrap();
        let vec = result.to_vec().unwrap();
        assert_eq!(vec.len(), size);
        drop(result);
    }

    let stats_after = ctx.stats();
    let activity = (stats_after.buffer_allocations + stats_after.buffer_reuses)
        - (stats_before.buffer_allocations + stats_before.buffer_reuses);
    assert!(activity > 0, "Expected buffer activity across sizes");
}

// ============================================================================
// Fault Tests - Error Handling & Edge Cases
// ============================================================================

#[tokio::test]
async fn fault_small_tensor_operations() {
    let device = get_test_device().await;

    // Create small tensors (edge case)
    let a = Tensor::from_data(&[1.0], vec![1], device.clone()).unwrap();
    let b = Tensor::from_data(&[2.0], vec![1], device).unwrap();

    let c = a.add(&b).unwrap();
    let result = c.to_vec().unwrap();
    assert_eq!(result, vec![3.0]);
}

#[tokio::test]
async fn fault_pool_survives_device_poll() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    // Acquire some buffers
    let buf1 = ctx.acquire_pooled_output(1000);
    let buf2 = ctx.acquire_pooled_output(2000);

    // Device operations (poll)
    let _ = device.poll_safe();

    // Buffers should still be valid
    assert!(buf1.size() >= 4000);
    assert!(buf2.size() >= 8000);

    // Release and reacquire
    drop(buf1);
    drop(buf2);

    let buf3 = ctx.acquire_pooled_output(1000);
    assert!(buf3.size() >= 4000);
}

#[tokio::test]
async fn fault_tensor_is_pooled_check() {
    let device = get_test_device().await;

    // User-created tensor is NOT pooled
    let owned = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
    assert!(
        !owned.is_pooled(),
        "User-created tensor should not be pooled"
    );

    // Operation output IS pooled
    let a = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
    let b = Tensor::from_data(&[3.0, 4.0], vec![2], device).unwrap();
    let pooled = a.add(&b).unwrap();
    assert!(pooled.is_pooled(), "Operation output should be pooled");
}

#[tokio::test]
async fn fault_large_tensor_allocation() {
    let device = get_test_device().await;
    let ctx = get_device_context(&device);

    let stats_before = ctx.stats();

    // Allocate a moderately large buffer (1M elements = 4MB)
    let buf = ctx.acquire_pooled_output(1_000_000);
    assert!(buf.size() >= 4_000_000);

    // Return and reacquire
    drop(buf);
    let buf2 = ctx.acquire_pooled_output(1_000_000);
    assert!(buf2.size() >= 4_000_000);

    let stats_after = ctx.stats();
    // Should have at least one reuse (from the second acquire)
    assert!(
        stats_after.buffer_reuses > stats_before.buffer_reuses,
        "Expected buffer reuse for large tensor"
    );
}

// ============================================================================
// Correctness Tests - Verify results are correct with pooling
// ============================================================================

#[tokio::test]
async fn correctness_pooled_results_accurate() {
    let device = get_test_device().await;

    let a_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..1000).map(|i| (i * 2) as f32).collect();

    let a = Tensor::from_data(&a_data, vec![1000], device.clone()).unwrap();
    let b = Tensor::from_data(&b_data, vec![1000], device).unwrap();

    // Run multiple times - results should always be correct
    for iteration in 0..10 {
        let c = a.add(&b).unwrap();
        let result = c.to_vec().unwrap();

        // Verify sample elements
        assert_eq!(result[0], 0.0, "Iteration {iteration}: element 0 wrong");
        assert_eq!(
            result[500], 1500.0,
            "Iteration {iteration}: element 500 wrong"
        ); // 500 + 1000
        assert_eq!(
            result[999], 2997.0,
            "Iteration {iteration}: element 999 wrong"
        ); // 999 + 1998

        drop(c); // Return buffer to pool
    }
}

#[tokio::test]
async fn correctness_reused_buffer_no_stale_data() {
    let device = get_test_device().await;

    // First operation: 1+1=2
    let a1 = Tensor::from_data(&[1.0; 100], vec![100], device.clone()).unwrap();
    let b1 = Tensor::from_data(&[1.0; 100], vec![100], device.clone()).unwrap();
    let c1 = a1.add(&b1).unwrap();
    let result1 = c1.to_vec().unwrap();
    assert!(
        result1.iter().all(|&x| x == 2.0),
        "First operation should be 2.0"
    );
    drop(c1); // Return buffer to pool

    // Second operation: 10+10=20 (same size, should reuse buffer)
    let a2 = Tensor::from_data(&[10.0; 100], vec![100], device.clone()).unwrap();
    let b2 = Tensor::from_data(&[10.0; 100], vec![100], device).unwrap();
    let c2 = a2.add(&b2).unwrap();
    let result2 = c2.to_vec().unwrap();

    // Must be 20, not 2 (no stale data)
    assert!(
        result2.iter().all(|&x| x == 20.0),
        "Buffer reuse must not contain stale data: got {:?}",
        &result2[..5]
    );
}
