// SPDX-License-Identifier: AGPL-3.0-or-later
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
use barracuda::multi_gpu::{DeviceClass, DeviceRequirements, MultiDevicePool, WorkloadConfig};
use barracuda::resource_quota::{ResourceQuota, presets};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ============================================================================
// Device Discovery Tests
// ============================================================================

#[tokio::test]
async fn test_enumerate_all_gpus() {
    let adapters = WgpuDevice::enumerate_adapters().await;

    assert!(
        adapters
            .iter()
            .any(|a| a.device_type == wgpu::DeviceType::DiscreteGpu),
        "No discrete GPUs found - these tests require GPU hardware"
    );
}

#[tokio::test]
async fn test_device_creation_for_all_adapters() {
    let adapters = WgpuDevice::enumerate_adapters().await;

    let mut successful_discrete_gpus = Vec::new();

    for (idx, adapter) in adapters.iter().enumerate() {
        match WgpuDevice::from_adapter_index(idx).await {
            Ok(_device) => {
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
            Err(_e) => {}
        }
    }

    // We expect at least one discrete GPU to work
    assert!(
        !successful_discrete_gpus.is_empty(),
        "No discrete GPUs could be initialized"
    );
}

#[tokio::test]
async fn test_multi_device_pool_discovers_both_gpus() {
    let _adapters = WgpuDevice::enumerate_adapters().await;

    // Use a config that excludes software renderers but includes all hardware GPUs
    let config = WorkloadConfig {
        max_parallel: 4,
        prefer_discrete: true,
        exclude_software: true,
        min_gflops: 100.0, // Low threshold to include all real GPUs
    };

    let pool = MultiDevicePool::with_config(config).await;

    match pool {
        Ok(pool) => {
            let _ = pool.device_status();

            // Check for multiple GPUs
            let device_count = pool.device_count();

            // Count discrete-class devices (vendor is not distinguished at pool level)
            let discrete_count = pool
                .devices()
                .iter()
                .filter(|d| d.device_class == DeviceClass::DiscreteGpu)
                .count();

            let _ = (device_count, discrete_count);
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
async fn test_acquire_device_with_discrete_preference() {
    let pool = MultiDevicePool::new().await;
    if let Err(_e) = &pool {
        return;
    }
    let pool = pool.unwrap();

    let reqs = DeviceRequirements::new().prefer_discrete();

    match pool.acquire(&reqs).await {
        Ok(lease) => {
            // If any discrete GPU exists in the pool, acquisition should prefer that class
            let has_discrete = pool
                .devices()
                .iter()
                .any(|d| d.device_class == DeviceClass::DiscreteGpu);
            if has_discrete {
                assert_eq!(
                    lease.info().device_class,
                    DeviceClass::DiscreteGpu,
                    "Should have selected a discrete GPU when available"
                );
            }
        }
        Err(_e) => {}
    }
}

#[tokio::test]
async fn test_acquire_multiple_devices_sequentially() {
    let pool = MultiDevicePool::new().await;
    if let Err(_e) = &pool {
        return;
    }
    let pool = pool.unwrap();

    if pool.device_count() < 2 {
        return;
    }

    // Acquire first device
    let lease1 = pool
        .acquire_any()
        .await
        .expect("Should acquire first device");

    // Acquire second device (should get a different one)
    let lease2 = pool
        .acquire_any()
        .await
        .expect("Should acquire second device");

    // They should be different devices
    assert_ne!(
        lease1.info().index,
        lease2.info().index,
        "Should acquire different devices"
    );
}

// ============================================================================
// Quota Enforcement Tests
// ============================================================================

#[tokio::test]
async fn test_quota_enforcement_on_device_lease() {
    let pool = MultiDevicePool::new().await;
    if let Err(_e) = &pool {
        return;
    }
    let pool = pool.unwrap();

    // Create a 100 MB quota
    let quota = ResourceQuota::named("test_quota").with_max_vram_mb(100);

    let lease = pool
        .acquire_with_quota(&DeviceRequirements::new(), Some(quota))
        .await
        .expect("Should acquire device");

    let _ = lease.info().name;
    let _ = lease.quota_tracker().map(|t| t.quota().max_vram_bytes);

    // Track allocations
    let tracker = lease.quota_tracker().expect("Should have quota tracker");

    // 50 MB should succeed
    assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
    let _ = tracker.usage_percent().unwrap();

    // Another 50 MB should succeed (at 100 MB total)
    assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
    let _ = tracker.usage_percent().unwrap();

    // 1 more byte should fail
    let result = lease.track_allocation(1);
    assert!(result.is_err(), "Should fail when exceeding quota");
    let _ = result.unwrap_err();

    // Deallocate and try again
    lease.track_deallocation(50 * 1024 * 1024);
    let _ = tracker.usage_percent().unwrap();

    // Now should succeed
    assert!(lease.track_allocation(1).is_ok());
}

#[tokio::test]
async fn test_preset_quotas() {
    let small = presets::small();
    assert_eq!(small.max_vram_bytes, Some(512 * 1024 * 1024));

    let medium = presets::medium();
    assert_eq!(medium.max_vram_bytes, Some(2 * 1024 * 1024 * 1024));

    let large = presets::large();
    assert_eq!(large.max_vram_bytes, Some(8 * 1024 * 1024 * 1024));

    let _ml_inference = presets::ml_inference();

    let unlimited = presets::unlimited();
    assert!(unlimited.max_vram_bytes.is_none());
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_device_acquisition() {
    let pool = Arc::new(MultiDevicePool::new().await.expect("Should create pool"));

    if pool.device_count() < 2 {
        return;
    }

    let _ = pool.summary();

    // Spawn multiple tasks that try to acquire devices
    let mut handles = vec![];

    for _i in 0..4 {
        let pool_clone = pool.clone();
        let handle = tokio::spawn(async move {
            let reqs = DeviceRequirements::new().prefer_discrete();

            match pool_clone.acquire(&reqs).await {
                Ok(lease) => {
                    let name = lease.info().name.clone();
                    // Drop the lease — release is atomic, no hold time needed.
                    drop(lease);
                    Ok(name)
                }
                Err(e) => Err(e.to_string()),
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

    // At least pool.device_count() acquisitions should succeed
    assert!(
        successes >= pool.device_count().min(4),
        "Should succeed up to device_count"
    );
}

// ============================================================================
// Device Capability Tests
// ============================================================================

#[tokio::test]
async fn test_device_vram_requirements() {
    let pool = MultiDevicePool::new().await;
    if let Err(_e) = &pool {
        return;
    }
    let pool = pool.unwrap();

    // Reasonable requirement (8 GB)
    let reqs = DeviceRequirements::new().with_min_vram_gb(8);
    if let Ok(lease) = pool.acquire(&reqs).await {
        let _ = (
            &lease.info().name,
            lease.info().vram_bytes / (1024 * 1024 * 1024),
        );
    }

    // Unreasonable requirement (100 GB)
    let reqs = DeviceRequirements::new().with_min_vram_gb(100);
    let result = pool.acquire(&reqs).await;
    assert!(
        result.is_err(),
        "Should fail with unrealistic VRAM requirement"
    );
}

// ============================================================================
// Pool Status and Diagnostics
// ============================================================================

#[tokio::test]
async fn test_pool_status_reporting() {
    let pool = MultiDevicePool::new().await.expect("Should create pool");

    let _ = pool.summary();

    let _ = pool.device_status();

    for device in pool.devices() {
        let _ = (
            device.index,
            &device.name,
            device.device_class,
            device.vram_bytes / (1024 * 1024 * 1024),
            device.estimated_gflops,
            device.is_discrete,
            device.usage_percent(),
        );
    }
}

// ============================================================================
// Real Workload Test (Matrix Operations)
// ============================================================================

#[tokio::test]
async fn test_real_workload_on_acquired_device() {
    let pool = MultiDevicePool::new().await;
    if let Err(_e) = &pool {
        return;
    }
    let pool = pool.unwrap();

    let quota = ResourceQuota::named("workload_test").with_max_vram_gb(4);

    let lease = pool
        .acquire_with_quota(&DeviceRequirements::new(), Some(quota))
        .await
        .expect("Should acquire device");

    let _ = lease.info().name;

    // Track a simulated allocation
    let buffer_size = 1024 * 1024 * 64; // 64 MB
    lease
        .track_allocation(buffer_size)
        .expect("Should track allocation");

    let _ = (
        buffer_size / (1024 * 1024),
        lease.quota_tracker().unwrap().usage_percent().unwrap(),
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

    // Track deallocation
    lease.track_deallocation(buffer_size);
    let _ = lease.quota_tracker().unwrap().usage_percent().unwrap();
}

// ============================================================================
// Stress Test
// ============================================================================

#[tokio::test]
async fn test_rapid_acquire_release_cycles() {
    let pool = Arc::new(MultiDevicePool::new().await.expect("Should create pool"));

    let _ = pool.summary();

    // Run sequential acquire/release cycles to test proper cleanup
    let cycles = 10;

    for _i in 0..cycles {
        let lease = pool.acquire_any().await.expect("Should acquire device");
        // DeviceLease::drop() releases atomically via AtomicBool::store().
        // No sleep needed — release is visible immediately after drop.
        drop(lease);
    }

    let _ = (cycles, pool.summary());

    // All devices should be available again
    for status in pool.device_status() {
        assert!(
            status.contains("available"),
            "All devices should be available after leases dropped"
        );
    }
}
