// SPDX-License-Identifier: AGPL-3.0-only
//! Multi-Device Pool Integration Tests
//!
//! Tests for `ResourceQuota` and `MultiDevicePool` with real GPU hardware.
//! Designed for heterogeneous GPU configurations (e.g., NVIDIA + AMD).
//!
//! # Test Environment
//!
//! These tests are designed to work with:
//! - 2+ discrete GPUs (e.g., RTX 3090 + RX 6950 XT)
//! - Mixed vendor configurations
//! - Single GPU fallback
//!
//! # Running Tests
//!
//! ```bash
//! cargo test -p barracuda --test multi_device_integration -- --nocapture
//! ```

#![expect(clippy::unwrap_used, reason = "tests")]
use barracuda::device::WgpuDevice;
use barracuda::multi_gpu::{DeviceRequirements, GpuVendor, MultiDevicePool, WorkloadConfig};
use barracuda::resource_quota::{ResourceQuota, presets};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ============================================================================
// Device Discovery Tests
// ============================================================================

#[tokio::test]
async fn test_enumerate_all_gpus() {
    let adapters = WgpuDevice::enumerate_adapters().await;

    println!("\n=== Available GPU Adapters ===");
    for adapter in &adapters {
        println!(
            "  - {} ({:?}, {:?})",
            adapter.name, adapter.device_type, adapter.backend
        );
    }

    // Filter to discrete GPUs only
    let discrete_gpus: Vec<_> = adapters
        .iter()
        .filter(|a| a.device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    println!("\n=== Discrete GPUs ===");
    for gpu in &discrete_gpus {
        println!("  - {}", gpu.name);
    }

    // We expect at least one discrete GPU for these tests
    assert!(
        !discrete_gpus.is_empty(),
        "No discrete GPUs found - these tests require GPU hardware"
    );
}

#[tokio::test]
async fn test_device_creation_for_all_adapters() {
    let adapters = WgpuDevice::enumerate_adapters().await;

    println!("\n=== Testing Device Creation for Each Adapter ===");

    let mut successful_discrete_gpus = Vec::new();

    for (idx, adapter) in adapters.iter().enumerate() {
        println!(
            "\n[{}] {} ({:?}, {:?})",
            idx, adapter.name, adapter.device_type, adapter.backend
        );

        match WgpuDevice::from_adapter_index(idx).await {
            Ok(device) => {
                println!("  ✓ Device created successfully");
                println!("    Queue family: {:?}", device.adapter_info().backend);
                // Count as discrete if wgpu says discrete OR if it's a known discrete GPU vendor
                // (some OpenGL adapters report as "Other" device type)
                let is_likely_discrete = adapter.device_type == wgpu::DeviceType::DiscreteGpu
                    || (adapter.device_type == wgpu::DeviceType::Other
                        && (adapter.name.to_lowercase().contains("nvidia")
                            || adapter.name.to_lowercase().contains("amd")
                            || adapter.name.to_lowercase().contains("radeon")));
                if is_likely_discrete {
                    successful_discrete_gpus.push(adapter.name.clone());
                }
            }
            Err(e) => {
                println!("  ✗ Failed: {e}");
            }
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Successfully created {} discrete GPU device(s):",
        successful_discrete_gpus.len()
    );
    for name in &successful_discrete_gpus {
        println!("  - {name}");
    }

    // We expect at least one discrete GPU to work
    assert!(
        !successful_discrete_gpus.is_empty(),
        "No discrete GPUs could be initialized"
    );
}

#[tokio::test]
async fn test_multi_device_pool_discovers_both_gpus() {
    // Print all available adapters first
    let adapters = WgpuDevice::enumerate_adapters().await;
    println!("\n=== All Adapters (before pool creation) ===");
    for (idx, adapter) in adapters.iter().enumerate() {
        println!(
            "  [{}] {} ({:?}, {:?})",
            idx, adapter.name, adapter.device_type, adapter.backend
        );
    }

    // Use a config that excludes software renderers but includes all hardware GPUs
    let config = WorkloadConfig {
        max_parallel: 4,
        prefer_discrete: true,
        exclude_software: true,
        min_gflops: 100.0, // Low threshold to include all real GPUs
    };

    println!("\n=== Creating MultiDevicePool ===");
    let pool = MultiDevicePool::with_config(config).await;

    match pool {
        Ok(pool) => {
            println!("\n=== MultiDevicePool Status ===");
            println!("{}", pool.summary());

            for status in pool.device_status() {
                println!("  {status}");
            }

            // Check for multiple GPUs
            let device_count = pool.device_count();
            println!("\nTotal devices: {device_count}");

            // Check for vendor diversity
            let has_nvidia = pool.devices().iter().any(|d| d.vendor == GpuVendor::Nvidia);
            let has_amd = pool.devices().iter().any(|d| d.vendor == GpuVendor::Amd);

            println!("Vendor coverage: NVIDIA={has_nvidia}, AMD={has_amd}");

            if device_count >= 2 {
                println!("✓ Multi-GPU configuration detected");
            } else {
                println!("⚠ Single GPU detected - multi-device tests will run in degraded mode");
            }
        }
        Err(e) => {
            panic!("Failed to create MultiDevicePool: {e}");
        }
    }
}

// ============================================================================
// Device Selection Tests
// ============================================================================

#[tokio::test]
async fn test_acquire_device_with_nvidia_preference() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    let reqs = DeviceRequirements::new().prefer_nvidia();

    match pool.acquire(&reqs).await {
        Ok(lease) => {
            println!("Acquired with NVIDIA preference: {}", lease.info().name);

            // If an NVIDIA GPU exists, we should get it
            let has_nvidia = pool.devices().iter().any(|d| d.vendor == GpuVendor::Nvidia);
            if has_nvidia {
                assert_eq!(
                    lease.info().vendor,
                    GpuVendor::Nvidia,
                    "Should have selected NVIDIA GPU when available"
                );
            }
        }
        Err(e) => {
            println!("Could not acquire device: {e}");
        }
    }
}

#[tokio::test]
async fn test_acquire_device_with_amd_preference() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    let reqs = DeviceRequirements::new().prefer_amd();

    match pool.acquire(&reqs).await {
        Ok(lease) => {
            println!("Acquired with AMD preference: {}", lease.info().name);

            // If an AMD GPU exists, we should get it
            let has_amd = pool.devices().iter().any(|d| d.vendor == GpuVendor::Amd);
            if has_amd {
                assert_eq!(
                    lease.info().vendor,
                    GpuVendor::Amd,
                    "Should have selected AMD GPU when available"
                );
            }
        }
        Err(e) => {
            println!("Could not acquire device: {e}");
        }
    }
}

#[tokio::test]
async fn test_acquire_multiple_devices_sequentially() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    if pool.device_count() < 2 {
        println!("Skipping - need 2+ GPUs for this test");
        return;
    }

    println!("\n=== Sequential Multi-Device Acquisition ===");

    // Acquire first device
    let lease1 = pool
        .acquire_any()
        .await
        .expect("Should acquire first device");
    println!("Device 1: {} ({})", lease1.info().name, lease1.info().index);

    // Acquire second device (should get a different one)
    let lease2 = pool
        .acquire_any()
        .await
        .expect("Should acquire second device");
    println!("Device 2: {} ({})", lease2.info().name, lease2.info().index);

    // They should be different devices
    assert_ne!(
        lease1.info().index,
        lease2.info().index,
        "Should acquire different devices"
    );

    println!("✓ Successfully acquired 2 different devices");
}

// ============================================================================
// Quota Enforcement Tests
// ============================================================================

#[tokio::test]
async fn test_quota_enforcement_on_device_lease() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    // Create a 100 MB quota
    let quota = ResourceQuota::named("test_quota").with_max_vram_mb(100);

    let lease = pool
        .acquire_with_quota(&DeviceRequirements::new(), Some(quota))
        .await
        .expect("Should acquire device");

    println!("\n=== Quota Enforcement Test ===");
    println!("Device: {}", lease.info().name);
    println!(
        "Quota: {:?}",
        lease.quota_tracker().map(|t| t.quota().max_vram_bytes)
    );

    // Track allocations
    let tracker = lease.quota_tracker().expect("Should have quota tracker");

    // 50 MB should succeed
    assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
    println!(
        "After 50 MB allocation: {:.1}% used",
        tracker.usage_percent().unwrap()
    );

    // Another 50 MB should succeed (at 100 MB total)
    assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
    println!(
        "After 100 MB allocation: {:.1}% used",
        tracker.usage_percent().unwrap()
    );

    // 1 more byte should fail
    let result = lease.track_allocation(1);
    assert!(result.is_err(), "Should fail when exceeding quota");
    println!("Quota exceeded as expected: {}", result.unwrap_err());

    // Deallocate and try again
    lease.track_deallocation(50 * 1024 * 1024);
    println!(
        "After deallocation: {:.1}% used",
        tracker.usage_percent().unwrap()
    );

    // Now should succeed
    assert!(lease.track_allocation(1).is_ok());
    println!("✓ Quota enforcement working correctly");
}

#[tokio::test]
async fn test_preset_quotas() {
    println!("\n=== Preset Quotas ===");

    let small = presets::small();
    println!("small: {:?} bytes", small.max_vram_bytes);
    assert_eq!(small.max_vram_bytes, Some(512 * 1024 * 1024));

    let medium = presets::medium();
    println!("medium: {:?} bytes", medium.max_vram_bytes);
    assert_eq!(medium.max_vram_bytes, Some(2 * 1024 * 1024 * 1024));

    let large = presets::large();
    println!("large: {:?} bytes", large.max_vram_bytes);
    assert_eq!(large.max_vram_bytes, Some(8 * 1024 * 1024 * 1024));

    let ml_inference = presets::ml_inference();
    println!(
        "ml_inference: {:?} bytes, {:?} buffers",
        ml_inference.max_vram_bytes, ml_inference.max_buffers
    );

    let unlimited = presets::unlimited();
    println!("unlimited: {:?}", unlimited.max_vram_bytes);
    assert!(unlimited.max_vram_bytes.is_none());

    println!("✓ All presets configured correctly");
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_device_acquisition() {
    let pool = Arc::new(MultiDevicePool::new().await.expect("Should create pool"));

    if pool.device_count() < 2 {
        println!("Skipping concurrent test - need 2+ GPUs");
        return;
    }

    println!("\n=== Concurrent Device Acquisition ===");
    println!("Pool: {}", pool.summary());

    // Spawn multiple tasks that try to acquire devices
    let mut handles = vec![];

    for i in 0..4 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            let reqs = if i % 2 == 0 {
                DeviceRequirements::new().prefer_nvidia()
            } else {
                DeviceRequirements::new().prefer_amd()
            };

            match pool_clone.acquire(&reqs).await {
                Ok(lease) => {
                    let name = lease.info().name.clone();
                    println!("Task {i} acquired: {name}");
                    // Drop the lease — release is atomic, no hold time needed.
                    drop(lease);
                    Ok(name)
                }
                Err(e) => {
                    println!("Task {i} failed: {e}");
                    Err(e.to_string())
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks
    let mut results = Vec::with_capacity(handles.len());
    for h in handles {
        results.push(h.await);
    }

    let successes = results.iter().filter(|r| r.is_ok()).count();
    println!("\n{}/{} acquisitions succeeded", successes, results.len());

    // At least pool.device_count() acquisitions should succeed
    assert!(
        successes >= pool.device_count().min(4),
        "Should succeed up to device_count"
    );

    println!("✓ Concurrent acquisition test passed");
}

// ============================================================================
// Device Capability Tests
// ============================================================================

#[tokio::test]
async fn test_device_vram_requirements() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    println!("\n=== VRAM Requirements Test ===");

    // Reasonable requirement (8 GB)
    let reqs = DeviceRequirements::new().with_min_vram_gb(8);
    match pool.acquire(&reqs).await {
        Ok(lease) => {
            println!(
                "Found device with 8+ GB: {} (~{} GB)",
                lease.info().name,
                lease.info().vram_bytes / (1024 * 1024 * 1024)
            );
        }
        Err(_) => {
            println!("No device with 8+ GB VRAM");
        }
    }

    // Unreasonable requirement (100 GB)
    let reqs = DeviceRequirements::new().with_min_vram_gb(100);
    let result = pool.acquire(&reqs).await;
    assert!(
        result.is_err(),
        "Should fail with unrealistic VRAM requirement"
    );
    println!("✓ Correctly rejected 100 GB VRAM requirement");
}

// ============================================================================
// Pool Status and Diagnostics
// ============================================================================

#[tokio::test]
async fn test_pool_status_reporting() {
    let pool = MultiDevicePool::new().await.expect("Should create pool");

    println!("\n=== Pool Status Report ===");
    println!("{}", pool.summary());

    println!("\nPer-device status:");
    for status in pool.device_status() {
        println!("  {status}");
    }

    println!("\nDetailed device info:");
    for device in pool.devices() {
        println!("  [{}] {}", device.index, device.name);
        println!("      Vendor: {:?}", device.vendor);
        println!(
            "      VRAM: {} GB (estimated)",
            device.vram_bytes / (1024 * 1024 * 1024)
        );
        println!("      GFLOPS: {:.0} (estimated)", device.estimated_gflops);
        println!("      Discrete: {}", device.is_discrete);
        println!("      Usage: {:.1}%", device.usage_percent());
    }

    println!("\n✓ Status reporting works correctly");
}

// ============================================================================
// Real Workload Test (Matrix Operations)
// ============================================================================

#[tokio::test]
async fn test_real_workload_on_acquired_device() {
    let pool = MultiDevicePool::new().await;
    if let Err(e) = &pool {
        println!("Skipping test - no GPU available: {e}");
        return;
    }
    let pool = pool.unwrap();

    let quota = ResourceQuota::named("workload_test").with_max_vram_gb(4);

    let lease = pool
        .acquire_with_quota(&DeviceRequirements::new(), Some(quota))
        .await
        .expect("Should acquire device");

    println!("\n=== Real Workload Test ===");
    println!("Device: {}", lease.info().name);

    // Track a simulated allocation
    let buffer_size = 1024 * 1024 * 64; // 64 MB
    lease
        .track_allocation(buffer_size)
        .expect("Should track allocation");

    println!(
        "Allocated {} MB, usage: {:.1}%",
        buffer_size / (1024 * 1024),
        lease.quota_tracker().unwrap().usage_percent().unwrap()
    );

    // Verify device is usable
    let device = lease.device();
    let _queue = device.queue();
    let wgpu_device = device.device();

    // Create a simple buffer to verify device works
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let _buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test_buffer"),
        contents: bytemuck::cast_slice(&test_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    println!("✓ Device is functional, buffer created successfully");

    // Track deallocation
    lease.track_deallocation(buffer_size);
    println!(
        "After deallocation, usage: {:.1}%",
        lease.quota_tracker().unwrap().usage_percent().unwrap()
    );
}

// ============================================================================
// Stress Test
// ============================================================================

#[tokio::test]
async fn test_rapid_acquire_release_cycles() {
    let pool = Arc::new(MultiDevicePool::new().await.expect("Should create pool"));

    println!("\n=== Rapid Acquire/Release Stress Test ===");
    println!("Initial: {}", pool.summary());

    // Run sequential acquire/release cycles to test proper cleanup
    let cycles = 10;

    for i in 0..cycles {
        let lease = pool.acquire_any().await.expect("Should acquire device");
        println!("Cycle {}: acquired {}", i, lease.info().name);
        // DeviceLease::drop() releases atomically via AtomicBool::store().
        // No sleep needed — release is visible immediately after drop.
        drop(lease);
    }

    println!("Completed {cycles} cycles");
    println!("Final: {}", pool.summary());

    // All devices should be available again
    for status in pool.device_status() {
        println!("  {status}");
        assert!(
            status.contains("available"),
            "All devices should be available after leases dropped"
        );
    }

    println!("✓ Stress test passed - all devices released correctly");
}
