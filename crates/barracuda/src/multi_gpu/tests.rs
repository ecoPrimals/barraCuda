// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-GPU module tests.

use super::strategy::{GpuPool, MultiDevicePool};
use super::topology::{GpuDriver, GpuVendor};
use super::types::{DeviceInfo, DeviceRequirements};
use crate::resource_quota::ResourceQuota;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

#[tokio::test]
async fn test_gpu_pool_creation() {
    let pool = GpuPool::new().await;
    if let Ok(pool) = pool {
        println!("Pool: {}", pool.summary());
        for device in pool.devices() {
            println!("  - {} ({:?})", device.name, device.vendor);
        }
    }
}

#[test]
fn test_vendor_detection() {
    assert_eq!(
        GpuVendor::from_name("NVIDIA GeForce RTX 3090"),
        GpuVendor::Nvidia
    );
    assert_eq!(
        GpuVendor::from_name("AMD Radeon RX 6950 XT (RADV NAVI21)"),
        GpuVendor::Amd
    );
    assert_eq!(GpuVendor::from_name("llvmpipe"), GpuVendor::Software);
}

#[tokio::test]
async fn test_multi_device_pool_creation() {
    let pool = MultiDevicePool::new().await;
    match pool {
        Ok(pool) => {
            println!("MultiDevicePool: {}", pool.summary());
            for status in pool.device_status() {
                println!("  {status}");
            }
        }
        Err(e) => {
            println!("No GPU available: {e}");
        }
    }
}

#[tokio::test]
async fn test_device_requirements() {
    let pool = MultiDevicePool::new().await;
    if let Ok(pool) = pool {
        let reqs = DeviceRequirements::new().prefer_nvidia();
        if let Ok(lease) = pool.acquire(&reqs).await {
            println!(
                "Acquired: {} ({:?})",
                lease.info().name,
                lease.info().vendor
            );
        }
        let reqs = DeviceRequirements::new().with_min_vram_gb(100);
        let result = pool.acquire(&reqs).await;
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_device_lease_tracking() {
    let pool = MultiDevicePool::new().await;
    if let Ok(pool) = pool {
        let quota = ResourceQuota::new().with_max_vram_mb(100);
        if let Ok(lease) = pool
            .acquire_with_quota(&DeviceRequirements::new(), Some(quota))
            .await
        {
            assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
            assert!(lease.track_allocation(50 * 1024 * 1024).is_ok());
            assert!(lease.track_allocation(1).is_err());
            lease.track_deallocation(50 * 1024 * 1024);
            assert!(lease.track_allocation(1).is_ok());
        }
    }
}

#[test]
fn test_device_requirements_scoring() {
    let reqs = DeviceRequirements::new()
        .prefer_nvidia()
        .with_min_vram_gb(8);

    let nvidia_info = DeviceInfo {
        index: 0,
        pool_index: 0,
        name: Arc::from("RTX 4070"),
        vendor: GpuVendor::Nvidia,
        driver: GpuDriver::NvidiaProprietary,
        vram_bytes: 12 * 1024 * 1024 * 1024,
        estimated_gflops: 5000.0,
        is_discrete: true,
        allocations: Arc::new(AtomicUsize::new(0)),
        allocated_bytes: Arc::new(AtomicU64::new(0)),
        busy: Arc::new(AtomicBool::new(false)),
    };

    let amd_info = DeviceInfo {
        index: 1,
        pool_index: 1,
        name: Arc::from("RX 6800"),
        vendor: GpuVendor::Amd,
        driver: GpuDriver::Radv,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        estimated_gflops: 4000.0,
        is_discrete: true,
        allocations: Arc::new(AtomicUsize::new(0)),
        allocated_bytes: Arc::new(AtomicU64::new(0)),
        busy: Arc::new(AtomicBool::new(false)),
    };

    let nvidia_score = reqs.score(&nvidia_info).unwrap();
    let amd_score = reqs.score(&amd_info).unwrap();
    assert!(nvidia_score > amd_score);
}
