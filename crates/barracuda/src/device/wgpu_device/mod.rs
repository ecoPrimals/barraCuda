// SPDX-License-Identifier: AGPL-3.0-only
//! Pure WGSL device - hardware-agnostic compute via WebGPU
//!
//! **Pure WGSL Architecture**:
//! - WGSL shaders ONLY (no separate CPU code!)
//! - wgpu handles execution on ANY device (GPU/CPU/NPU/TPU)
//!
//! ## Adapter Selection
//!
//! Set `BARRACUDA_GPU_ADAPTER` environment variable:
//! - `BARRACUDA_GPU_ADAPTER=0` — Select first adapter
//! - `BARRACUDA_GPU_ADAPTER=titan` — Select adapter containing "titan"
//! - `BARRACUDA_GPU_ADAPTER=auto` — Use wgpu `HighPerformance` (default)

mod buffers;
mod capabilities;
mod compilation;
mod creation;
mod dispatch;
mod guard;

pub(crate) use dispatch::{DispatchPermit, DispatchSemaphore, concurrency_budget};
pub use guard::{GuardedDeviceHandle, GuardedEncoder};

use super::autotune::{GLOBAL_TUNER, GpuCalibration, GpuDeviceForCalibration};
use crate::error::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Maximum time to wait for a single GPU poll to complete.
///
/// Configurable via `BARRACUDA_POLL_TIMEOUT_SECS`. Defaults to 120 seconds,
/// which is generous for any real GPU operation. This prevents indefinite
/// hangs under instrumentation (llvm-cov) or when a software rasterizer
/// stalls. The CPU sends work to the GPU and expects results within this
/// window; exceeding it signals a driver-level stall, not a compute issue.
pub(crate) fn poll_timeout() -> Option<Duration> {
    static TIMEOUT: std::sync::LazyLock<Option<Duration>> = std::sync::LazyLock::new(|| {
        let secs: u64 = std::env::var("BARRACUDA_POLL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        if secs == 0 {
            None
        } else {
            Some(Duration::from_secs(secs))
        }
    });
    *TIMEOUT
}

/// Timeout for the dispatch semaphore before falling back to a blocking acquire.
/// Override with `BARRACUDA_DISPATCH_TIMEOUT_SECS` (default: 30).
fn dispatch_semaphore_timeout() -> Duration {
    static TIMEOUT: std::sync::LazyLock<Duration> = std::sync::LazyLock::new(|| {
        let secs: u64 = std::env::var("BARRACUDA_DISPATCH_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
        Duration::from_secs(secs)
    });
    *TIMEOUT
}

/// WebGPU device - executes WGSL on any hardware
///
/// Concurrency model (atomic barrier + Mutex):
/// - `active_encoders` is an `AtomicU32` counting live `GuardedEncoder`s.
///   Encoding is **never blocked** — just an atomic increment on creation
///   and decrement on finish/drop.
/// - `gpu_lock` is a `Mutex<()>` that serializes submit and poll against
///   each other. Poll additionally spin-waits until `active_encoders == 0`
///   to prevent wgpu-core storage races between poll cleanup and encoding.
/// - `dispatch_semaphore` limits how many operations can be in-flight
///   simultaneously based on device type (prevents driver overload).
#[derive(Debug, Clone)]
pub struct WgpuDevice {
    pub(crate) device: GuardedDeviceHandle,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    calibration: Option<GpuCalibration>,
    /// Vulkan pipeline cache — avoids re-compiling identical SPIR-V to machine code.
    pipeline_cache: Option<Arc<wgpu::PipelineCache>>,
    /// Set when the GPU reports a device-lost error.
    pub(crate) lost: Arc<AtomicBool>,
    /// Serializes submit+poll to prevent wgpu-core storage races.
    gpu_lock: Arc<std::sync::Mutex<()>>,
    /// Counts live `GuardedEncoders` — poll waits for this to reach zero.
    active_encoders: Arc<std::sync::atomic::AtomicU32>,
    /// Limits concurrent dispatches based on device capability.
    dispatch_semaphore: Arc<DispatchSemaphore>,
    /// Optional VRAM quota tracker — when set, buffer allocations are checked
    /// against the budget before proceeding. `None` means unlimited.
    quota_tracker: Option<Arc<crate::resource_quota::QuotaTracker>>,
}

impl GpuDeviceForCalibration for WgpuDevice {
    fn device(&self) -> &wgpu::Device {
        &self.device
    }

    fn name(&self) -> &str {
        &self.adapter_info.name
    }

    fn submit_and_poll_calibration(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.submit_and_poll_inner(commands);
    }
}

impl WgpuDevice {
    /// Get device name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get device type
    #[must_use]
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.adapter_info.device_type
    }

    /// Check if running on CPU fallback
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        self.adapter_info.device_type == wgpu::DeviceType::Cpu
    }

    /// Check if the GPU driver has reported this device as lost.
    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Attach a VRAM quota tracker to this device.
    ///
    /// Once set, all buffer allocations through the canonical helpers
    /// (`create_buffer_f32`, `create_buffer_f64`, etc.) will be checked
    /// against the budget before proceeding.
    pub fn set_quota_tracker(&mut self, tracker: Arc<crate::resource_quota::QuotaTracker>) {
        self.quota_tracker = Some(tracker);
    }

    /// Current quota tracker, if any.
    #[must_use]
    pub fn quota_tracker(&self) -> Option<&Arc<crate::resource_quota::QuotaTracker>> {
        self.quota_tracker.as_ref()
    }

    /// Check the VRAM quota and record the allocation. No-op if no tracker is set.
    pub(crate) fn quota_try_allocate(&self, bytes: u64) -> Result<()> {
        if let Some(tracker) = &self.quota_tracker {
            tracker.try_allocate(bytes)?;
        }
        Ok(())
    }

    /// Record a buffer deallocation against the VRAM quota. No-op if no tracker is set.
    ///
    /// Callers should invoke this when a tracked buffer is dropped or returned
    /// to a pool. Currently called by `DeviceLease` and by buffer-drop hooks
    /// when a `QuotaTracker` is attached.
    pub fn quota_deallocate(&self, bytes: u64) {
        if let Some(tracker) = &self.quota_tracker {
            tracker.deallocate(bytes);
        }
    }

    /// Check if f64 shaders are truly available on this device.
    /// Returns `true` only when BOTH conditions hold:
    /// 1. `wgpu::Features::SHADER_F64` was granted at device creation.
    /// 2. The runtime probe (if completed) confirms basic f64 compilation works.
    ///    groundSpring V37 discovered that NVK/NAK advertise `SHADER_F64` but fail
    ///    to compile even basic f64 WGSL. This method consults the probe cache to
    ///    avoid silent shader compilation failures.
    #[must_use]
    pub fn has_f64_shaders(&self) -> bool {
        if !self.device.features().contains(wgpu::Features::SHADER_F64) {
            return false;
        }
        super::probe::cached_f64_builtins(self).is_none_or(|caps| caps.can_compile_f64())
    }

    /// Check if the Sovereign Compiler's SPIR-V passthrough path is available.
    /// In wgpu 28 SPIR-V passthrough is gated by the `spirv` cargo feature
    /// on the wgpu crate (always enabled in our workspace). The Vulkan
    /// backend is required at runtime for the driver to accept SPIR-V.
    #[must_use]
    pub fn has_spirv_passthrough(&self) -> bool {
        self.adapter_info.backend == wgpu::Backend::Vulkan
    }

    /// Access underlying wgpu device
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get a clone of the wgpu device (cheap — internally Arc-wrapped in wgpu 28).
    #[must_use]
    pub fn device_clone(&self) -> wgpu::Device {
        (*self.device).clone()
    }

    /// Get the shared GPU lock (for components that call `device.poll` directly).
    pub(crate) fn gpu_lock_arc(&self) -> Arc<std::sync::Mutex<()>> {
        self.gpu_lock.clone()
    }

    /// Get the active encoder counter (for `BufferPool` poll coordination).
    pub(crate) fn active_encoders_arc(&self) -> Arc<std::sync::atomic::AtomicU32> {
        self.active_encoders.clone()
    }

    /// Increment active encoder count (encoding phase barrier).
    /// While any encoder is active, `device.poll()` will spin-wait.
    /// This is lock-free and never blocks the caller.
    pub fn encoding_guard(&self) {
        self.active_encoders.fetch_add(1, Ordering::Acquire);
    }

    /// Decrement active encoder count (encoding phase complete).
    pub fn encoding_complete(&self) {
        self.active_encoders.fetch_sub(1, Ordering::Release);
    }

    /// Bounded wait for active encoders to finish.
    /// Encoding is CPU work (recording commands into a buffer), so it
    /// completes in microseconds. A brief yield loop avoids the
    /// wgpu-core race where poll/cleanup overlaps with encoding, without
    /// risking starvation even at 128+ threads.
    pub(crate) fn brief_encoder_wait(&self) {
        for _ in 0..256 {
            if self.active_encoders.load(Ordering::Acquire) == 0 {
                return;
            }
            std::thread::yield_now();
        }
    }

    /// Create a command encoder protected by the active-encoder barrier.
    /// The returned `GuardedEncoder` increments the active encoder count,
    /// preventing `device.poll()` from running. Encoding is lock-free and
    /// never blocks other threads. Call `.finish()` to release.
    #[must_use]
    pub fn create_encoder_guarded(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> GuardedEncoder {
        self.active_encoders.fetch_add(1, Ordering::Acquire);
        let encoder = self.device.create_command_encoder(desc);
        GuardedEncoder::new(encoder, self.active_encoders.clone())
    }

    /// Get adapter info (for capability detection)
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Get the device's pipeline cache for `create_compute_pipeline` calls.
    /// Returns `None` only when the Vulkan driver does not support pipeline caching.
    #[must_use]
    pub fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.pipeline_cache.as_deref()
    }

    /// Access command queue
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get a clone of the command queue (cheap — internally Arc-wrapped in wgpu 28).
    #[must_use]
    pub fn queue_clone(&self) -> wgpu::Queue {
        self.queue.clone()
    }

    /// Acquire a dispatch permit from the device's concurrency budget.
    /// Hold this for the entire compile→bind→submit lifecycle of an
    /// operation. The device automatically sizes the budget to what the
    /// hardware can handle (2 for CPU/llvmpipe, 8 for discrete GPU, etc.).
    #[must_use]
    pub fn acquire_dispatch(&self) -> DispatchPermit<'_> {
        self.dispatch_semaphore.acquire()
    }

    /// Poll the device for completed work, catching device-lost panics.
    /// Every `device.poll(PollType::Wait)` call site in barracuda should
    /// go through this method (or its `_nonblocking` variant) so that a
    /// driver-level device loss is consistently caught, flagged, and
    /// converted to `Err` instead of panicking the caller's thread.
    /// Serialized via `gpu_lock`: wgpu-core's internal resource tracking
    /// on software rasterizers can race under concurrent submit/poll.
    /// # Errors
    /// Returns [`Err`] if the device is lost or poll fails (e.g. driver crash).
    pub fn poll_safe(&self) -> Result<()> {
        if self.is_lost() {
            return Err(crate::error::BarracudaError::device_lost("device lost"));
        }
        let _guard = self
            .gpu_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.brief_encoder_wait();
        let device = self.device.clone();
        let timeout = poll_timeout();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            match device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout,
            }) {
                Ok(status) => status.is_queue_empty(),
                Err(_) => false,
            }
        }));
        match result {
            Ok(true) => Ok(()),
            Ok(false) => {
                tracing::warn!(
                    "poll_safe: GPU poll timed out after {}s — treating as stall",
                    timeout.map_or(0, |d| d.as_secs())
                );
                Err(crate::error::BarracudaError::execution_failed(
                    "GPU poll timed out — driver stall or instrumentation overhead",
                ))
            }
            Err(payload) => {
                if Self::is_device_lost_panic(&payload) {
                    self.lost.store(true, Ordering::Release);
                    tracing::warn!("poll_safe: device lost: {}", Self::panic_message(&payload));
                    return Err(crate::error::BarracudaError::device_lost("device lost"));
                }
                tracing::warn!(
                    "poll_safe: wgpu panic (non-fatal): {}",
                    Self::panic_message(&payload)
                );
                Ok(())
            }
        }
    }

    /// Submit command buffers without polling for completion.
    /// Serialized via `gpu_lock` to prevent wgpu-core storage races on
    /// software rasterizers. Use when the caller will poll separately
    /// (e.g. via `poll_safe()` or `map_staging_buffer()`).
    pub fn submit_commands(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        if self.is_lost() {
            return;
        }
        let _guard = self
            .gpu_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.brief_encoder_wait();
        let queue = self.queue.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            queue.submit(commands);
        }));
        if let Err(payload) = result {
            if Self::is_device_lost_panic(&payload) {
                self.lost.store(true, Ordering::Release);
                tracing::warn!(
                    "submit_commands: device lost (flagged): {}",
                    Self::panic_message(&payload)
                );
                return;
            }
            std::panic::resume_unwind(payload);
        }
    }

    /// Non-blocking poll variant for drain/housekeeping paths.
    /// Serialized via `gpu_lock` to prevent wgpu-core cleanup races.
    pub fn poll_nonblocking(&self) {
        if self.is_lost() {
            return;
        }
        if self.active_encoders.load(Ordering::Acquire) > 0 {
            return;
        }
        let Ok(_guard) = self.gpu_lock.try_lock() else {
            return;
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.device.poll(wgpu::PollType::Poll);
        }));
    }

    /// Extract a human-readable message from a `catch_unwind` payload.
    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
        payload
            .downcast_ref::<String>()
            .map(std::string::String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown")
    }

    fn is_device_lost_panic(payload: &Box<dyn std::any::Any + Send>) -> bool {
        let msg = Self::panic_message(payload);
        msg.contains("lost") || msg.contains("Lost") || msg.contains("Parent device")
    }

    /// Submit commands and poll, respecting the device's concurrency budget.
    /// First attempts a timed acquire (30 s) — if the semaphore is
    /// saturated beyond that, falls back to the blocking path with a
    /// warning.  This provides observability into dispatch stalls without
    /// changing correctness: the operation always completes.
    pub fn submit_and_poll(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        let timeout = dispatch_semaphore_timeout();
        let _permit = if let Some(p) = self.dispatch_semaphore.try_acquire_timeout(timeout) {
            p
        } else {
            tracing::warn!(
                "dispatch semaphore saturated for {}s ({} permits) — blocking until available",
                timeout.as_secs(),
                self.dispatch_semaphore.max_permits(),
            );
            self.dispatch_semaphore.acquire()
        };
        self.submit_and_poll_inner(commands);
    }

    /// Submit without acquiring a dispatch permit.
    /// Use when the caller already holds a `DispatchPermit`.
    /// Submit and poll use SEPARATE write-lock acquisitions. This allows
    /// other threads to interleave their submits between our submit and poll,
    /// dramatically reducing lock-convoy stalls on software rasterizers.
    /// poll(Wait) processes ALL pending work, so interleaved submits from
    /// other threads are completed too.
    pub(crate) fn submit_and_poll_inner(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        if self.is_lost() {
            return;
        }
        let _guard = self
            .gpu_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.brief_encoder_wait();
        {
            let queue = self.queue.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                queue.submit(commands);
            }));
            if let Err(payload) = result {
                if Self::is_device_lost_panic(&payload) {
                    self.lost.store(true, Ordering::Release);
                    tracing::warn!(
                        "submit_and_poll: device lost during submit: {}",
                        Self::panic_message(&payload)
                    );
                    return;
                }
                std::panic::resume_unwind(payload);
            }
        }
        self.brief_encoder_wait();
        {
            let device = self.device.clone();
            let timeout = poll_timeout();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let queue_empty = match device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout,
                }) {
                    Ok(status) => status.is_queue_empty(),
                    Err(_) => false,
                };
                if !queue_empty {
                    tracing::warn!(
                        "submit_and_poll: GPU poll timed out after {}s",
                        timeout.map_or(0, |d| d.as_secs())
                    );
                }
            }));
            if let Err(payload) = result {
                if Self::is_device_lost_panic(&payload) {
                    self.lost.store(true, Ordering::Release);
                    tracing::warn!(
                        "submit_and_poll: device lost during poll: {}",
                        Self::panic_message(&payload)
                    );
                    return;
                }
                std::panic::resume_unwind(payload);
            }
        }
    }

    /// The device's concurrency budget (max simultaneous dispatches).
    #[must_use]
    pub fn max_concurrent_dispatches(&self) -> usize {
        self.dispatch_semaphore.max_permits()
    }

    /// Execute WGSL compute shader
    /// Acquires a dispatch permit for the full compile→submit lifecycle.
    /// Holds encoding read-guard during compile/encode, releases before submit.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, pipeline creation fails,
    /// or GPU dispatch fails (e.g., device lost).
    pub fn execute_compute(
        &self,
        shader_source: &str,
        bind_groups: &[&wgpu::BindGroup],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let _permit = self.acquire_dispatch();
        let commands = {
            self.encoding_guard();
            let shader = self.compile_shader(shader_source, Some("barraCuda Operation"));
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("barraCuda Pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    cache: self.pipeline_cache(),
                    compilation_options: Default::default(),
                });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("barraCuda Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("barraCuda Compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                for (i, bg) in bind_groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, Some(*bg), &[]);
                }
                pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            }

            let cmds = encoder.finish();
            self.encoding_complete();
            cmds
        };
        self.submit_and_poll_inner(Some(commands));
        Ok(())
    }

    /// Get calibration for this device
    pub fn get_calibration(&self) -> GpuCalibration {
        GLOBAL_TUNER.get_or_calibrate(self)
    }

    /// Force recalibration
    pub fn recalibrate(&self) -> GpuCalibration {
        GLOBAL_TUNER.recalibrate(self)
    }

    /// Get optimal workgroup size for this device
    #[must_use]
    pub fn optimal_workgroup_size(&self) -> u32 {
        self.calibration.as_ref().map_or_else(
            || GLOBAL_TUNER.get_or_calibrate(self).optimal_workgroup_size,
            |c| c.optimal_workgroup_size,
        )
    }

    /// Get measured peak bandwidth for this device (GB/s)
    #[must_use]
    pub fn peak_bandwidth_gbps(&self) -> f64 {
        self.get_calibration().peak_bandwidth_gbps
    }

    /// Get measured dispatch overhead for this device (μs)
    #[must_use]
    pub fn dispatch_overhead_us(&self) -> f64 {
        self.get_calibration().dispatch_overhead_us
    }

    /// Create calibrated device (runs calibration immediately)
    /// # Errors
    /// Returns [`Err`] if no WGPU adapter is found or device creation fails.
    pub async fn new_calibrated() -> Result<Self> {
        let mut device = Self::new().await?;
        let cal = device.get_calibration();
        device.calibration = Some(cal);
        Ok(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgpu_device_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        println!("barraCuda device: {}", device.name());
        if device.is_cpu() {
            println!("  Using CPU software rasterizer");
        } else {
            println!("  Using GPU acceleration");
        }
    }

    #[tokio::test]
    async fn test_enumerate_adapters() {
        let adapters = WgpuDevice::enumerate_adapters().await;
        assert!(
            !adapters.is_empty(),
            "WGPU should find at least one adapter"
        );
    }

    #[tokio::test]
    async fn test_buffer_operations() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let buffer = device.create_buffer_f32(10).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        device.write_buffer_f32(&buffer, &data).unwrap();
        let read_data = device.read_buffer_f32(&buffer, 10).unwrap();
        assert_eq!(read_data, data);
    }

    #[tokio::test]
    async fn test_adapter_selector_auto() {
        let _ = WgpuDevice::with_adapter_selector("auto").await;
    }

    #[tokio::test]
    async fn test_adapter_selector_index() {
        let adapters = WgpuDevice::enumerate_adapters().await;
        if adapters.is_empty() {
            return;
        }
        if let Ok(device) = WgpuDevice::with_adapter_selector("0").await {
            assert_eq!(device.name(), adapters[0].name);
        }
    }

    #[tokio::test]
    async fn test_adapter_selector_name_match() {
        let adapters = WgpuDevice::enumerate_adapters().await;
        if adapters.is_empty() {
            return;
        }
        let partial = adapters[0]
            .name
            .chars()
            .take(4)
            .collect::<String>()
            .to_lowercase();
        let _ = WgpuDevice::with_adapter_selector(&partial).await;
    }

    #[tokio::test]
    async fn test_adapter_selector_fallthrough() {
        let adapters = WgpuDevice::enumerate_adapters().await;
        let large_index = (adapters.len() + 1000).to_string();
        if let Err(e) = WgpuDevice::with_adapter_selector(&large_index).await {
            assert!(e.to_string().contains("No adapter matches"));
        }
    }

    #[tokio::test]
    async fn test_from_env_default() {
        // Tests the "auto" selection path — same code as from_env() when
        // BARRACUDA_GPU_ADAPTER is unset (the typical case).
        let _ = WgpuDevice::with_adapter_selector("auto").await;
    }

    #[tokio::test]
    async fn test_driver_detection_apis() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let _ = device.is_nvk();
        let _ = device.is_radv();
        let _ = device.is_nvidia_proprietary();
    }

    #[test]
    fn test_driver_detection_logic() {
        fn contains_nvk_markers(driver: &str) -> bool {
            let lower = driver.to_lowercase();
            lower.contains("nvk") || lower.contains("nouveau") || lower.contains("mesa")
        }
        fn contains_radv_markers(driver: &str) -> bool {
            driver.to_lowercase().contains("radv")
        }
        assert!(contains_nvk_markers("NVK"));
        assert!(contains_nvk_markers("nouveau"));
        assert!(!contains_nvk_markers("NVIDIA"));
        assert!(contains_radv_markers("RADV"));
        assert!(!contains_radv_markers("NVIDIA"));
    }
}
