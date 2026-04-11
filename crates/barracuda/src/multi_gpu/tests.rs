// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-GPU module tests.

use super::strategy::{GpuPool, MultiDevicePool};
use super::topology::DeviceClass;
use super::types::{DeviceInfo, DeviceRequirements};
use crate::resource_quota::ResourceQuota;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};

#[tokio::test]
async fn test_gpu_pool_creation() {
    let pool = GpuPool::new().await;
    if let Ok(pool) = pool {
        let _ = pool.summary();
        for _ in pool.devices() {}
    }
}

#[test]
fn test_device_class_detection() {
    assert_eq!(
        DeviceClass::from_device_type(wgpu::DeviceType::DiscreteGpu, "NVIDIA GeForce RTX 3090"),
        DeviceClass::DiscreteGpu
    );
    assert_eq!(
        DeviceClass::from_device_type(
            wgpu::DeviceType::DiscreteGpu,
            "AMD Radeon RX 6950 XT (RADV NAVI21)"
        ),
        DeviceClass::DiscreteGpu
    );
    assert_eq!(
        DeviceClass::from_device_type(wgpu::DeviceType::Cpu, "llvmpipe"),
        DeviceClass::Software
    );
}

#[tokio::test]
async fn test_multi_device_pool_creation() {
    let pool = MultiDevicePool::new().await;
    match pool {
        Ok(pool) => {
            let _ = pool.summary();
            for status in pool.device_status() {
                let _ = status;
            }
        }
        Err(e) => {
            let _ = e;
        }
    }
}

#[tokio::test]
async fn test_device_requirements() {
    let pool = MultiDevicePool::new().await;
    if let Ok(pool) = pool {
        let reqs = DeviceRequirements::new().prefer_discrete();
        let _ = pool.acquire(&reqs).await;
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
        .prefer_discrete()
        .with_min_vram_gb(8);

    let discrete_info = DeviceInfo {
        index: 0,
        pool_index: 0,
        name: Arc::from("RTX 4070"),
        device_class: DeviceClass::DiscreteGpu,
        vram_bytes: 12 * 1024 * 1024 * 1024,
        estimated_gflops: 5000.0,
        is_discrete: true,
        f64_builtins_available: true,
        allocations: Arc::new(AtomicUsize::new(0)),
        allocated_bytes: Arc::new(AtomicU64::new(0)),
        busy: Arc::new(AtomicBool::new(false)),
    };

    let integrated_info = DeviceInfo {
        index: 1,
        pool_index: 1,
        name: Arc::from("Integrated GPU"),
        device_class: DeviceClass::IntegratedGpu,
        vram_bytes: 16 * 1024 * 1024 * 1024,
        estimated_gflops: 4000.0,
        is_discrete: false,
        f64_builtins_available: true,
        allocations: Arc::new(AtomicUsize::new(0)),
        allocated_bytes: Arc::new(AtomicU64::new(0)),
        busy: Arc::new(AtomicBool::new(false)),
    };

    let discrete_score = reqs.score(&discrete_info).unwrap();
    let integrated_score = reqs.score(&integrated_info);
    // Integrated is excluded because require_discrete is not set, but
    // discrete should score higher due to prefer_discrete class bonus
    assert!(
        discrete_score > integrated_score.unwrap_or(0),
        "Discrete ({discrete_score}) should score higher"
    );
}
