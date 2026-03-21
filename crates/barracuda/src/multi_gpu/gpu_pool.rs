// SPDX-License-Identifier: AGPL-3.0-or-later
//! Basic GPU pool with round-robin workload routing and semaphore-based concurrency.

use super::topology::{DeviceClass, GpuInfo, WorkloadType};
use super::types::WorkloadConfig;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Pool of GPU devices for parallel execution.
pub struct GpuPool {
    devices: Vec<Arc<WgpuDevice>>,
    info: Vec<GpuInfo>,
    semaphore: Arc<Semaphore>,
}

impl GpuPool {
    /// Create a pool with default workload config.
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

        for (idx, adapter) in adapters.iter().enumerate() {
            let device_class = DeviceClass::from_device_type(adapter.device_type, &adapter.name);
            if config.exclude_software && device_class == DeviceClass::Software {
                continue;
            }

            let gflops = super::estimate_gflops(device_class, adapter.device_type);

            if gflops < config.min_gflops {
                continue;
            }

            if let Ok(device) = WgpuDevice::from_adapter_index(idx).await {
                let f64_builtins = !device.needs_f64_exp_log_workaround();
                info.push(GpuInfo {
                    index: idx,
                    name: adapter.name.clone(),
                    device_class,
                    gflops,
                    busy: false,
                    f64_builtins_available: f64_builtins,
                });
                devices.push(Arc::new(device));
            }
        }

        let mut indices: Vec<usize> = (0..devices.len()).collect();
        indices.sort_by(|&a, &b| {
            info[b]
                .gflops
                .partial_cmp(&info[a].gflops)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_devices: Vec<_> = indices.iter().map(|&i| devices[i].clone()).collect();
        let sorted_info: Vec<_> = indices.iter().map(|&i| info[i].clone()).collect();

        tracing::info!("GPU pool initialized with {} devices", sorted_devices.len());
        for gi in &sorted_info {
            tracing::info!(
                "  - {} ({:?}, ~{:.0} GFLOPS)",
                gi.name,
                gi.device_class,
                gi.gflops
            );
        }

        let max_parallel = config.max_parallel.min(sorted_devices.len()).max(1);

        Ok(Self {
            devices: sorted_devices,
            info: sorted_info,
            semaphore: Arc::new(Semaphore::new(max_parallel)),
        })
    }

    /// Number of devices in the pool.
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Device metadata for all devices in the pool.
    #[must_use]
    pub fn devices(&self) -> &[GpuInfo] {
        &self.info
    }

    /// Get device by index.
    #[must_use]
    pub fn device(&self, index: usize) -> Option<Arc<WgpuDevice>> {
        self.devices.get(index).cloned()
    }

    /// Route a workload to the best available device.
    pub fn route(&self, workload: WorkloadType) -> Option<(Arc<WgpuDevice>, GpuInfo)> {
        if self.devices.is_empty() {
            return None;
        }
        match workload {
            WorkloadType::Streaming => self
                .devices
                .first()
                .map(|d| (d.clone(), self.info[0].clone())),
            WorkloadType::Iterative => {
                for (i, gi) in self.info.iter().enumerate() {
                    if gi.is_compute_capable() && gi.supports_f64_builtins() {
                        return Some((self.devices[i].clone(), self.info[i].clone()));
                    }
                }
                self.devices
                    .first()
                    .map(|d| (d.clone(), self.info[0].clone()))
            }
            WorkloadType::F64Builtins => {
                for (i, gi) in self.info.iter().enumerate() {
                    if gi.supports_f64_builtins() {
                        let d: Arc<WgpuDevice> = Arc::clone(&self.devices[i]);
                        return Some((d, self.info[i].clone()));
                    }
                }
                tracing::warn!("No GPU with f64 builtin support found - workload may fail on NVK");
                self.devices
                    .first()
                    .map(|d| (d.clone(), self.info[0].clone()))
            }
        }
    }

    /// Acquire a device for the workload, holding a semaphore permit.
    /// # Errors
    /// Returns [`Err`] if semaphore acquisition fails or no GPU is available for
    /// the workload type.
    pub async fn route_acquire(
        &self,
        workload: WorkloadType,
    ) -> Result<(Arc<WgpuDevice>, GpuInfo, tokio::sync::OwnedSemaphorePermit)> {
        let permit = Arc::clone(&self.semaphore)
            .acquire_owned()
            .await
            .map_err(|e| BarracudaError::device(format!("Semaphore error: {e}")))?;

        let (device, info) = self
            .route(workload)
            .ok_or_else(|| BarracudaError::device_not_found("No GPU available for workload"))?;

        Ok((device, info, permit))
    }

    /// Execute a function on the first available device.
    /// # Errors
    /// Returns [`Err`] if semaphore acquisition fails, no GPU is available, or the
    /// spawned task panics or the closure returns an error.
    pub async fn execute<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(Arc<WgpuDevice>) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| BarracudaError::device(format!("Semaphore error: {e}")))?;

        let device = self
            .devices
            .first()
            .cloned()
            .ok_or_else(|| BarracudaError::device_not_found("No GPU available"))?;

        tokio::task::spawn_blocking(move || f(device))
            .await
            .map_err(|e| BarracudaError::device(format!("Task error: {e}")))?
    }

    /// Map data across devices in parallel (round-robin).
    /// # Errors
    /// Returns [`Err`] if any element's closure returns an error, a spawned task
    /// panics, or a task join fails.
    pub async fn parallel_map<T, R, F>(&self, data: Vec<T>, f: F) -> Result<Vec<R>>
    where
        T: Send + 'static,
        R: Send + 'static,
        F: Fn(Arc<WgpuDevice>, T) -> Result<R> + Send + Sync + Clone + 'static,
    {
        let num_devices = self.devices.len().max(1);
        let mut handles = Vec::new();

        for (i, chunk) in data.into_iter().enumerate() {
            let device = self.devices[i % num_devices].clone();
            let f = f.clone();
            let semaphore = self.semaphore.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await;
                tokio::task::spawn_blocking(move || f(device, chunk)).await
            });

            handles.push(handle);
        }

        let mut results = Vec::with_capacity(handles.len());
        for h in handles {
            results.push(h.await);
        }
        let mut output = Vec::new();
        for result in results {
            match result {
                Ok(Ok(Ok(value))) => output.push(value),
                Ok(Ok(Err(e))) => return Err(e),
                Ok(Err(e)) => return Err(BarracudaError::device(format!("Task error: {e}"))),
                Err(e) => return Err(BarracudaError::device(format!("Join error: {e}"))),
            }
        }
        Ok(output)
    }

    /// Human-readable pool summary.
    #[must_use]
    pub fn summary(&self) -> String {
        let total_gflops: f64 = self.info.iter().map(|g| g.gflops).sum();
        let discrete_count = self
            .info
            .iter()
            .filter(|g| g.device_class == DeviceClass::DiscreteGpu)
            .count();

        format!(
            "{} GPUs ({} discrete), ~{:.0} GFLOPS total",
            self.devices.len(),
            discrete_count,
            total_gflops
        )
    }
}
