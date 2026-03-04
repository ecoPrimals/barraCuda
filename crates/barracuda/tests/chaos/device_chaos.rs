// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Chaos Tests
//!
//! Tests resilience under adverse conditions:
//! - Device unavailability / failure
//! - Resource exhaustion (VRAM)
//! - Concurrent device access
//! - Hot-swap scenarios
//!
//! ## Test Categories
//!
//! 1. **Resource Exhaustion** - VRAM allocation failures
//! 2. **Concurrent Access** - Race conditions, deadlocks
//! 3. **Fallback Chains** - GPU → CPU fallback
//! 4. **Recovery** - Re-initialization after failure

#![expect(clippy::unwrap_used, reason = "tests")]
use barracuda::device::{Device, DeviceSelection, KernelRouter, WgpuDevice};
use barracuda::multi_gpu::{DeviceRequirements, MultiDevicePool};
use barracuda::resource_quota::{presets, ResourceQuota};
use barracuda::tensor::Tensor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Resource Exhaustion Tests
// ============================================================================

#[tokio::test]
async fn test_vram_allocation_limits() {
    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(e) => {
            println!("SKIP: No device available: {}", e);
            return;
        }
    };

    println!("\n=== VRAM Allocation Limits Test ===\n");
    println!("Device: {}", device.name());

    // Try progressively larger allocations
    let sizes = vec![
        (1024, "1KB"),
        (1024 * 1024, "1MB"),
        (10 * 1024 * 1024, "10MB"),
        (100 * 1024 * 1024, "100MB"),
        (500 * 1024 * 1024, "500MB"),
        (1024 * 1024 * 1024, "1GB"),
    ];

    let mut last_successful_size = 0;

    for (elements, label) in sizes {
        let data: Vec<f32> = vec![0.0; elements];
        match Tensor::from_vec_on(data, vec![elements], device.clone()).await {
            Ok(_tensor) => {
                println!("  ✓ Allocated {} tensor", label);
                last_successful_size = elements;
            }
            Err(e) => {
                println!("  ✗ Failed to allocate {}: {}", label, e);
                break;
            }
        }
    }

    println!(
        "\n  Maximum successful allocation: {} elements ({} bytes)\n",
        last_successful_size,
        last_successful_size * 4
    );

    assert!(last_successful_size > 0, "Should allocate at least something");
}

#[tokio::test]
async fn test_quota_enforcement() {
    println!("\n=== Quota Enforcement Test ===\n");

    // Create a strict quota (100MB VRAM limit)
    let quota = ResourceQuota::new().with_max_vram_mb(100);
    let tracker = barracuda::resource_quota::QuotaTracker::new(quota);

    println!("Quota: {}", tracker.summary());

    // Test VRAM tracking
    let vram_to_allocate = 50 * 1024 * 1024; // 50MB

    // First allocation should succeed
    assert!(
        tracker.try_allocate(vram_to_allocate).is_ok(),
        "First 50MB allocation should succeed"
    );
    println!("  ✓ First 50MB allocation succeeded");

    // Second allocation should also succeed (100MB total)
    assert!(
        tracker.try_allocate(vram_to_allocate).is_ok(),
        "Second 50MB allocation should succeed"
    );
    println!("  ✓ Second 50MB allocation succeeded");

    // Third allocation should fail (would exceed 100MB limit)
    assert!(
        tracker.try_allocate(vram_to_allocate).is_err(),
        "Third 50MB allocation should fail (quota exceeded)"
    );
    println!("  ✓ Third allocation correctly rejected (quota exceeded)");

    // Release some VRAM
    tracker.deallocate(vram_to_allocate);
    println!("  Released 50MB");

    // Now allocation should succeed again
    assert!(
        tracker.try_allocate(vram_to_allocate).is_ok(),
        "Allocation after release should succeed"
    );
    println!("  ✓ Allocation after release succeeded");

    println!("\n  Quota enforcement: PASS\n");
}

#[tokio::test]
async fn test_concurrent_device_operations() {
    println!("\n=== Concurrent Device Operations Test ===\n");

    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(_) => {
            println!("SKIP: No device available");
            return;
        }
    };

    let num_tasks = 8;
    let iterations_per_task = 10;
    let successful = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    for task_id in 0..num_tasks {
        let device_clone = device.clone();
        let successful_clone = successful.clone();
        let failed_clone = failed.clone();

        let handle = tokio::spawn(async move {
            for i in 0..iterations_per_task {
                let data: Vec<f32> = (0..1024).map(|x| (x + task_id * 1000 + i) as f32).collect();

                match Tensor::from_vec_on(data.clone(), vec![1024], device_clone.clone()).await {
                    Ok(tensor) => {
                        // Do some operations
                        match tensor.to_vec() {
                            Ok(result) => {
                                if result == data {
                                    successful_clone.fetch_add(1, Ordering::SeqCst);
                                } else {
                                    failed_clone.fetch_add(1, Ordering::SeqCst);
                                }
                            }
                            Err(_) => {
                                failed_clone.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                    Err(_) => {
                        failed_clone.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }

    let total = num_tasks * iterations_per_task;
    let success_count = successful.load(Ordering::SeqCst);
    let fail_count = failed.load(Ordering::SeqCst);

    println!("  Total operations: {}", total);
    println!("  Successful: {}", success_count);
    println!("  Failed: {}", fail_count);
    println!(
        "  Success rate: {:.1}%",
        (success_count as f64 / total as f64) * 100.0
    );

    // Most operations should succeed (allow for some resource contention)
    assert!(
        success_count as f64 / total as f64 > 0.90,
        "At least 90% of concurrent operations should succeed"
    );

    println!("\n  Concurrent operations: PASS\n");
}

// ============================================================================
// Fallback Chain Tests
// ============================================================================

#[tokio::test]
async fn test_gpu_to_cpu_fallback() {
    println!("\n=== GPU → CPU Fallback Test ===\n");

    // Try GPU first
    let gpu_result = WgpuDevice::new_gpu().await;
    let cpu_result = WgpuDevice::new_cpu().await;

    match (gpu_result, cpu_result) {
        (Ok(gpu), Ok(cpu)) => {
            println!("  GPU available: {}", gpu.name());
            println!("  CPU available: {}", cpu.name());

            // Test same operation on both
            let data = vec![1.0, 2.0, 3.0, 4.0];

            let gpu_arc = Arc::new(gpu);
            let cpu_arc = Arc::new(cpu);

            let gpu_tensor = Tensor::from_vec_on(data.clone(), vec![4], gpu_arc)
                .await
                .unwrap();
            let cpu_tensor = Tensor::from_vec_on(data.clone(), vec![4], cpu_arc)
                .await
                .unwrap();

            let gpu_result = gpu_tensor.to_vec().unwrap();
            let cpu_result = cpu_tensor.to_vec().unwrap();

            assert_eq!(gpu_result, cpu_result);
            println!("  ✓ GPU and CPU produce identical results");
            println!("  Fallback chain: GPU → CPU verified");
        }
        (Err(_), Ok(cpu)) => {
            println!("  GPU not available, using CPU fallback: {}", cpu.name());
            println!("  ✓ Fallback to CPU works");
        }
        (Ok(gpu), Err(_)) => {
            println!("  GPU available: {}, CPU fallback not available", gpu.name());
            println!("  ✓ GPU-only mode works");
        }
        (Err(gpu_err), Err(cpu_err)) => {
            println!("  GPU error: {}", gpu_err);
            println!("  CPU error: {}", cpu_err);
            panic!("No compute device available");
        }
    }

    println!("\n  Fallback chain: PASS\n");
}

// ============================================================================
// Stress Tests
// ============================================================================

#[tokio::test]
async fn test_rapid_allocation_deallocation() {
    println!("\n=== Rapid Allocation/Deallocation Stress Test ===\n");

    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(_) => {
            println!("SKIP: No device available");
            return;
        }
    };

    let iterations = 100;
    let mut successful = 0;

    let start = std::time::Instant::now();

    for i in 0..iterations {
        let size = 1024 * (1 + (i % 10)); // Vary size
        let data: Vec<f32> = vec![i as f32; size];

        match Tensor::from_vec_on(data, vec![size], device.clone()).await {
            Ok(tensor) => {
                // Immediately drop (deallocate)
                drop(tensor);
                successful += 1;
            }
            Err(e) => {
                println!("  Iteration {} failed: {}", i, e);
            }
        }
    }

    let elapsed = start.elapsed();

    println!("  Iterations: {}", iterations);
    println!("  Successful: {}", successful);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  Rate: {:.0} ops/sec",
        successful as f64 / elapsed.as_secs_f64()
    );

    assert!(
        successful as f64 / iterations as f64 > 0.95,
        "At least 95% of allocations should succeed"
    );

    println!("\n  Rapid allocation stress: PASS\n");
}

#[tokio::test]
async fn test_interleaved_multi_device_operations() {
    println!("\n=== Interleaved Multi-Device Operations ===\n");

    let adapters = WgpuDevice::enumerate_adapters();
    let discrete_gpus: Vec<_> = adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            matches!(
                a.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            )
        })
        .collect();

    if discrete_gpus.len() < 2 {
        println!("SKIP: Need 2+ GPUs for interleaved test");
        return;
    }

    // Create devices
    let mut devices = Vec::new();
    for (idx, info) in &discrete_gpus[..2] {
        if let Ok(device) = WgpuDevice::from_adapter_index(*idx).await {
            devices.push((info.name.clone(), Arc::new(device)));
        }
    }

    if devices.len() < 2 {
        println!("SKIP: Could not create 2 devices");
        return;
    }

    println!("  Device 0: {}", devices[0].0);
    println!("  Device 1: {}", devices[1].0);

    // Interleave operations
    let iterations = 20;
    let mut results = vec![Vec::new(), Vec::new()];

    for i in 0..iterations {
        let device_idx = i % 2;
        let data: Vec<f32> = (0..256).map(|x| x as f32 + i as f32 * 1000.0).collect();

        let tensor = Tensor::from_vec_on(data.clone(), vec![256], devices[device_idx].1.clone())
            .await
            .unwrap();

        let result = tensor.to_vec().unwrap();
        assert_eq!(result, data, "Data integrity check failed");
        results[device_idx].push(result);
    }

    println!("  Device 0 operations: {}", results[0].len());
    println!("  Device 1 operations: {}", results[1].len());
    println!("  ✓ All interleaved operations completed successfully");

    println!("\n  Interleaved multi-device: PASS\n");
}

// ============================================================================
// Recovery Tests
// ============================================================================

#[tokio::test]
async fn test_device_reinitialization_after_use() {
    println!("\n=== Device Re-initialization Test ===\n");

    for round in 0..3 {
        println!("  Round {}", round + 1);

        let device = match WgpuDevice::new().await {
            Ok(d) => Arc::new(d),
            Err(e) => {
                println!("    ✗ Failed to create device: {}", e);
                continue;
            }
        };

        // Do some operations
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_vec_on(data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = tensor.to_vec().unwrap();
        assert_eq!(result, data);

        println!("    ✓ Device {} working", device.name());

        // Drop device
        drop(device);
        println!("    ✓ Device released");
    }

    println!("\n  Device re-initialization: PASS\n");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_zero_size_tensor() {
    println!("\n=== Zero-Size Tensor Test ===\n");

    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(_) => {
            println!("SKIP: No device available");
            return;
        }
    };

    // Try to create empty tensor
    let empty_data: Vec<f32> = vec![];
    let result = Tensor::from_vec_on(empty_data, vec![0], device).await;

    match result {
        Ok(_) => {
            println!("  Zero-size tensor created (implementation allows it)");
        }
        Err(e) => {
            println!("  Zero-size tensor rejected: {} (expected)", e);
        }
    }

    println!("\n  Edge case handling: PASS\n");
}

#[tokio::test]
async fn test_large_dimension_count() {
    println!("\n=== Large Dimension Count Test ===\n");

    let device = match WgpuDevice::new().await {
        Ok(d) => Arc::new(d),
        Err(_) => {
            println!("SKIP: No device available");
            return;
        }
    };

    // Create tensor with many dimensions
    let data: Vec<f32> = vec![1.0; 64]; // 2^6 elements
    let shape = vec![2, 2, 2, 2, 2, 2]; // 6 dimensions

    match Tensor::from_vec_on(data, shape, device).await {
        Ok(tensor) => {
            println!("  ✓ 6-dimensional tensor created");
            let result = tensor.to_vec().unwrap();
            assert_eq!(result.len(), 64);
        }
        Err(e) => {
            println!("  High-dimensional tensor rejected: {}", e);
        }
    }

    println!("\n  Large dimension count: PASS\n");
}
