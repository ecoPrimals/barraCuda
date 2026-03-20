// SPDX-License-Identifier: AGPL-3.0-or-later
//! Advanced multi-device pool with requirements-based selection and resource quotas.

use super::{BYTES_PER_GIB, estimate_gflops, estimate_vram_bytes};

use super::topology::{GpuDriver, GpuVendor};
use super::types::{DeviceInfo, DeviceRequirements, WorkloadConfig};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::resource_quota::{QuotaTracker, ResourceQuota};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::{Mutex, Semaphore};

struct MultiDevicePoolInner {
    devices: Vec<Arc<WgpuDevice>>,
    info: Vec<DeviceInfo>,
    semaphore: Arc<Semaphore>,
    device_busy: Vec<Arc<std::sync::atomic::AtomicBool>>,
    selection_lock: Mutex<()>,
}

impl MultiDevicePoolInner {
    fn release_device(&self, index: usize) {
        if let Some(busy) = self.device_busy.get(index) {
            busy.store(false, Ordering::Release);
        }
    }
}

/// Lease of a device from the multi-device pool; releases on drop.
pub struct DeviceLease {
    device: Arc<WgpuDevice>,
    info: DeviceInfo,
    pool: Arc<MultiDevicePoolInner>,
    quota_tracker: Option<Arc<QuotaTracker>>,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl DeviceLease {
    /// The leased device.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Device metadata.
    #[must_use]
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Optional quota tracker for allocation limits.
    #[must_use]
    pub fn quota_tracker(&self) -> Option<&Arc<QuotaTracker>> {
        self.quota_tracker.as_ref()
    }

    /// Track an allocation against quota and device stats.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the configured resource quota.
    pub fn track_allocation(&self, bytes: u64) -> Result<()> {
        if let Some(tracker) = &self.quota_tracker {
            tracker.try_allocate(bytes)?;
        }
        self.info
            .allocated_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.info.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Track a deallocation.
    pub fn track_deallocation(&self, bytes: u64) {
        if let Some(tracker) = &self.quota_tracker {
            tracker.deallocate(bytes);
        }
        self.info
            .allocated_bytes
            .fetch_sub(bytes, Ordering::Relaxed);
        self.info.allocations.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        self.pool.release_device(self.info.pool_index);
    }
}

/// Advanced device pool with quotas and requirements-based selection.
///
/// # NVK device-creation serialization (hotSpring S68)
///
/// Device creation MUST remain sequential (not `join_all`). On NVK (nouveau),
/// concurrent `wgpu::Adapter::request_device` calls race on the kernel DRM
/// file descriptor and can trigger a mesa assertion or silent device loss.
/// The current `for` loop in `with_config` serializes creation by design.
pub struct MultiDevicePool {
    inner: Arc<MultiDevicePoolInner>,
}

impl MultiDevicePool {
    /// Create a pool with default config.
    /// # Errors
    /// Returns [`Err`] if no suitable GPU devices are found or device creation fails.
    pub async fn new() -> Result<Self> {
        Self::with_config(WorkloadConfig::default()).await
    }

    /// Create a pool with the given workload config.
    /// # Errors
    /// Returns [`Err`] if no suitable GPU devices are found (e.g. all adapters fail
    /// creation, none meet `min_gflops`, or all are excluded as software).
    pub async fn with_config(config: WorkloadConfig) -> Result<Self> {
        let adapters = WgpuDevice::enumerate_adapters().await;
        let mut devices = Vec::new();
        let mut info = Vec::new();
        let mut device_busy = Vec::new();

        for (idx, adapter) in adapters.iter().enumerate() {
            let vendor = GpuVendor::from_name(&adapter.name);
            if config.exclude_software && vendor == GpuVendor::Software {
                continue;
            }

            let is_likely_discrete = adapter.device_type == wgpu::DeviceType::DiscreteGpu
                || (adapter.device_type == wgpu::DeviceType::Other
                    && (vendor == GpuVendor::Nvidia || vendor == GpuVendor::Amd));

            let estimated_gflops = estimate_gflops(vendor, adapter.device_type);
            let estimated_vram = estimate_vram_bytes(vendor, adapter.device_type);

            if estimated_gflops < config.min_gflops {
                continue;
            }

            tracing::debug!(
                "Attempting to create device for adapter {}: {} ({:?})",
                idx,
                adapter.name,
                adapter.device_type
            );

            match WgpuDevice::from_adapter_index(idx).await {
                Ok(device) => {
                    tracing::info!(
                        "Successfully created device for adapter {}: {}",
                        idx,
                        adapter.name
                    );
                    let busy = Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let allocations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                    let allocated_bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));

                    let driver = GpuDriver::from_adapter_info(
                        &adapter.name,
                        &adapter.driver,
                        &adapter.driver_info,
                    );

                    info.push(DeviceInfo {
                        index: idx,
                        pool_index: 0,
                        name: Arc::from(adapter.name.as_str()),
                        vendor,
                        driver,
                        vram_bytes: estimated_vram,
                        estimated_gflops,
                        is_discrete: is_likely_discrete,
                        allocations: allocations.clone(),
                        allocated_bytes: allocated_bytes.clone(),
                        busy: busy.clone(),
                    });
                    device_busy.push(busy);
                    devices.push(Arc::new(device));
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to create device for adapter {}: {} - {}",
                        idx,
                        adapter.name,
                        e
                    );
                }
            }
        }

        if devices.is_empty() {
            return Err(BarracudaError::device_not_found(
                "No suitable GPU devices found",
            ));
        }

        let mut indices: Vec<usize> = (0..devices.len()).collect();
        indices.sort_by(|&a, &b| {
            info[b]
                .estimated_gflops
                .partial_cmp(&info[a].estimated_gflops)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_devices: Vec<_> = indices.iter().map(|&i| devices[i].clone()).collect();
        let mut sorted_info: Vec<_> = indices.iter().map(|&i| info[i].clone()).collect();
        let sorted_busy: Vec<_> = indices.iter().map(|&i| device_busy[i].clone()).collect();

        for (pool_idx, di) in sorted_info.iter_mut().enumerate() {
            di.pool_index = pool_idx;
        }

        tracing::info!(
            "MultiDevicePool initialized with {} devices",
            sorted_devices.len()
        );
        for di in &sorted_info {
            tracing::info!(
                "  - {} ({:?}, ~{:.0} GFLOPS, ~{} GB VRAM)",
                di.name,
                di.vendor,
                di.estimated_gflops,
                di.vram_bytes / BYTES_PER_GIB
            );
        }

        let max_parallel = config.max_parallel.min(sorted_devices.len()).max(1);

        Ok(Self {
            inner: Arc::new(MultiDevicePoolInner {
                devices: sorted_devices,
                info: sorted_info,
                semaphore: Arc::new(Semaphore::new(max_parallel)),
                device_busy: sorted_busy,
                selection_lock: Mutex::new(()),
            }),
        })
    }

    /// Number of devices in the pool.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.inner.devices.len()
    }

    /// Device metadata for all devices.
    #[must_use]
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.inner.info
    }

    /// Acquire a device matching the requirements.
    /// # Errors
    /// Returns [`Err`] if the semaphore acquisition fails (e.g. pool closed) or no
    /// device matches the requirements.
    pub async fn acquire(&self, requirements: &DeviceRequirements) -> Result<DeviceLease> {
        self.acquire_with_quota(requirements, None).await
    }

    /// Acquire with an optional resource quota.
    /// # Errors
    /// Returns [`Err`] if the semaphore acquisition fails (e.g. pool closed) or no
    /// device matches the requirements.
    pub async fn acquire_with_quota(
        &self,
        requirements: &DeviceRequirements,
        quota: Option<ResourceQuota>,
    ) -> Result<DeviceLease> {
        let permit = self
            .inner
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| BarracudaError::device(format!("Semaphore error: {e}")))?;

        let _lock = self.inner.selection_lock.lock().await;

        let mut best_idx = None;
        let mut best_score = i64::MIN;

        for (i, info) in self.inner.info.iter().enumerate() {
            if self.inner.device_busy[i].load(Ordering::Acquire) {
                continue;
            }
            if let Some(score) = requirements.score(info) {
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
        }

        let idx = best_idx
            .ok_or_else(|| BarracudaError::device_not_found("No device matches requirements"))?;

        self.inner.device_busy[idx].store(true, Ordering::Release);

        let quota_tracker = quota.map(|q| Arc::new(QuotaTracker::new(q)));

        Ok(DeviceLease {
            device: self.inner.devices[idx].clone(),
            info: self.inner.info[idx].clone(),
            pool: self.inner.clone(),
            quota_tracker,
            _permit: permit,
        })
    }

    /// Acquire any available device.
    /// # Errors
    /// Returns [`Err`] if the semaphore acquisition fails or no device is available.
    pub async fn acquire_any(&self) -> Result<DeviceLease> {
        self.acquire(&DeviceRequirements::new()).await
    }

    /// Get device by index.
    #[must_use]
    pub fn device(&self, index: usize) -> Option<Arc<WgpuDevice>> {
        self.inner.devices.get(index).cloned()
    }

    /// Execute a function on a device matching requirements.
    /// # Errors
    /// Returns [`Err`] if device acquisition fails (semaphore or no match), or if the
    /// spawned task panics or the closure returns an error.
    pub async fn execute<F, T>(&self, requirements: &DeviceRequirements, f: F) -> Result<T>
    where
        F: FnOnce(Arc<WgpuDevice>) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let lease = self.acquire(requirements).await?;
        let device = lease.device().clone();

        tokio::task::spawn_blocking(move || f(device))
            .await
            .map_err(|e| BarracudaError::device(format!("Task error: {e}")))?
    }

    /// Human-readable pool summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let total_vram: u64 = self.inner.info.iter().map(|d| d.vram_bytes).sum();
        let allocated_vram: u64 = self
            .inner
            .info
            .iter()
            .map(super::types::DeviceInfo::allocated_bytes)
            .sum();
        let total_gflops: f64 = self.inner.info.iter().map(|d| d.estimated_gflops).sum();

        let nvidia_count = self
            .inner
            .info
            .iter()
            .filter(|d| d.vendor == GpuVendor::Nvidia)
            .count();
        let amd_count = self
            .inner
            .info
            .iter()
            .filter(|d| d.vendor == GpuVendor::Amd)
            .count();
        let busy_count = self
            .inner
            .device_busy
            .iter()
            .filter(|b| b.load(Ordering::Relaxed))
            .count();

        format!(
            "{} GPUs ({} NVIDIA, {} AMD), ~{:.0} GFLOPS, ~{} GB total VRAM ({} GB allocated), {} busy",
            self.inner.devices.len(),
            nvidia_count,
            amd_count,
            total_gflops,
            total_vram / BYTES_PER_GIB,
            allocated_vram / BYTES_PER_GIB,
            busy_count
        )
    }

    /// Per-device status strings for diagnostics.
    #[must_use]
    pub fn device_status(&self) -> Vec<String> {
        self.inner
            .info
            .iter()
            .enumerate()
            .map(|(i, info)| {
                let busy = self.inner.device_busy[i].load(Ordering::Relaxed);
                format!(
                    "[{}] {} ({:?}): {:.1}% used, {} allocations, {}",
                    i,
                    info.name,
                    info.vendor,
                    info.usage_percent(),
                    info.allocation_count(),
                    if busy { "BUSY" } else { "available" }
                )
            })
            .collect()
    }
}
