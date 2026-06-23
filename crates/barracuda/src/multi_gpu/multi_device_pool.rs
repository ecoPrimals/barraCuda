// SPDX-License-Identifier: AGPL-3.0-or-later
//! Advanced multi-device pool with requirements-based selection and resource quotas.

use super::{BYTES_PER_GIB, estimate_gflops, estimate_vram_bytes};

use super::topology::DeviceClass;
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
            let device_class = DeviceClass::from_device_type(adapter.device_type, &adapter.name);
            if config.exclude_software && device_class == DeviceClass::Software {
                continue;
            }

            let is_likely_discrete = adapter.device_type == wgpu::DeviceType::DiscreteGpu
                || (adapter.device_type == wgpu::DeviceType::Other
                    && device_class == DeviceClass::DiscreteGpu);

            let estimated_gflops = estimate_gflops(device_class, adapter.device_type);
            let estimated_vram = estimate_vram_bytes(device_class, adapter.device_type);

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

                    let f64_builtins = !device.needs_f64_exp_log_workaround();

                    info.push(DeviceInfo {
                        index: idx,
                        pool_index: 0,
                        name: Arc::from(adapter.name.as_str()),
                        device_class,
                        vram_bytes: estimated_vram,
                        estimated_gflops,
                        is_discrete: is_likely_discrete,
                        f64_builtins_available: f64_builtins,
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
                di.device_class,
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
            if let Some(score) = requirements.score(info)
                && score > best_score
            {
                best_score = score;
                best_idx = Some(i);
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

    /// Execute with automatic OOM migration across pool devices.
    ///
    /// If the closure returns an OOM error (detected via [`BarracudaError::is_oom`]),
    /// the failed device's OOM flag is set, and the workload is retried on the next
    /// available device (up to `max_retries` attempts). This enables transparent
    /// VRAM pressure handling in multi-GPU configurations.
    ///
    /// The closure factory `make_f` is called for each attempt (since the closure
    /// is consumed on execution). It receives the attempt index (0-based).
    ///
    /// # Errors
    /// Returns the last OOM error if all devices are exhausted, or any non-OOM error
    /// immediately (no retry for logic/validation errors).
    pub async fn execute_with_migration<F, T>(
        &self,
        requirements: &DeviceRequirements,
        max_retries: usize,
        make_f: impl Fn(usize) -> F,
    ) -> Result<T>
    where
        F: FnOnce(Arc<WgpuDevice>) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        self.execute_with_migration_quota(requirements, max_retries, None, make_f)
            .await
    }

    /// Execute with automatic OOM migration and optional quota enforcement.
    ///
    /// Like [`execute_with_migration`](Self::execute_with_migration), but each
    /// device lease carries the provided [`ResourceQuota`]. When OOM is detected
    /// the quota tracker records the failure before migrating, giving downstream
    /// code visibility into which devices hit pressure.
    ///
    /// # Errors
    /// Returns the last OOM error if all devices are exhausted, or any non-OOM error
    /// immediately (no retry for logic/validation errors).
    pub async fn execute_with_migration_quota<F, T>(
        &self,
        requirements: &DeviceRequirements,
        max_retries: usize,
        quota: Option<ResourceQuota>,
        make_f: impl Fn(usize) -> F,
    ) -> Result<T>
    where
        F: FnOnce(Arc<WgpuDevice>) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let mut excluded: Vec<usize> = Vec::new();
        let max_attempts = max_retries.saturating_add(1).min(self.device_count());
        let mut last_error = None;
        let quota_tracker = quota.map(|q| Arc::new(QuotaTracker::new(q)));

        for attempt in 0..max_attempts {
            let lease = self
                .acquire_excluding(requirements, &excluded, quota_tracker.clone())
                .await;
            let lease = match lease {
                Ok(l) => l,
                Err(e) => {
                    tracing::debug!(
                        attempt,
                        "OOM migration: no more devices available — returning last error"
                    );
                    return Err(last_error.unwrap_or(e));
                }
            };

            let pool_index = lease.info().pool_index;
            let device_name = lease.info().name.clone();
            let device = lease.device().clone();

            let f = make_f(attempt);
            let result = tokio::task::spawn_blocking(move || f(device))
                .await
                .map_err(|e| BarracudaError::device(format!("Task error: {e}")))?;

            match result {
                Ok(v) => return Ok(v),
                Err(e) if e.is_oom() => {
                    tracing::warn!(
                        attempt,
                        device = %device_name,
                        "OOM detected — marking device and migrating workload"
                    );
                    if let Some(dev) = self.inner.devices.get(pool_index) {
                        dev.set_oom();
                    }
                    if let Some(ref tracker) = quota_tracker {
                        tracker.record_oom_failure();
                    }
                    excluded.push(pool_index);
                    last_error = Some(e);
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_error.unwrap_or_else(|| {
            BarracudaError::device("OOM migration exhausted all available devices")
        }))
    }

    /// Acquire a device excluding specific indices, with optional quota tracker.
    ///
    /// Also skips devices that have their OOM flag set, preventing re-acquisition
    /// of devices under memory pressure during migration retries.
    async fn acquire_excluding(
        &self,
        requirements: &DeviceRequirements,
        excluded: &[usize],
        quota_tracker: Option<Arc<QuotaTracker>>,
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
            if excluded.contains(&i) {
                continue;
            }
            if self.inner.device_busy[i].load(Ordering::Acquire) {
                continue;
            }
            if self.inner.devices[i].is_oom() {
                continue;
            }
            if let Some(score) = requirements.score(info)
                && score > best_score
            {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let idx = best_idx.ok_or_else(|| {
            BarracudaError::device_not_found(
                "No device matches requirements (excluded OOM devices)",
            )
        })?;

        self.inner.device_busy[idx].store(true, Ordering::Release);

        Ok(DeviceLease {
            device: self.inner.devices[idx].clone(),
            info: self.inner.info[idx].clone(),
            pool: self.inner.clone(),
            quota_tracker,
            _permit: permit,
        })
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

        let discrete_count = self
            .inner
            .info
            .iter()
            .filter(|d| d.device_class == DeviceClass::DiscreteGpu)
            .count();
        let busy_count = self
            .inner
            .device_busy
            .iter()
            .filter(|b| b.load(Ordering::Relaxed))
            .count();

        format!(
            "{} GPUs ({} discrete), ~{:.0} GFLOPS, ~{} GB total VRAM ({} GB allocated), {} busy",
            self.inner.devices.len(),
            discrete_count,
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
                    info.device_class,
                    info.usage_percent(),
                    info.allocation_count(),
                    if busy { "BUSY" } else { "available" }
                )
            })
            .collect()
    }
}
