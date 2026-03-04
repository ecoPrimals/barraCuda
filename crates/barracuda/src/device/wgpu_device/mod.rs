// SPDX-License-Identifier: AGPL-3.0-or-later
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
//! - `BARRACUDA_GPU_ADAPTER=auto` — Use wgpu HighPerformance (default)

mod buffers;
mod capabilities;
mod compilation;
mod creation;
mod dispatch;

pub(crate) use dispatch::{concurrency_budget, DispatchPermit, DispatchSemaphore};

use super::autotune::{GpuCalibration, GpuDeviceForCalibration, GLOBAL_TUNER};
use crate::error::Result;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// RAII command encoder that prevents `device.poll()` from running.
///
/// While this encoder exists, an atomic counter is non-zero, which causes
/// poll to spin-wait. Multiple `GuardedEncoder`s can coexist with zero
/// contention (just atomic increments). Encoding is never blocked.
///
/// Call `.finish()` to consume the encoder, decrement the counter, and
/// obtain a `wgpu::CommandBuffer` ready for submission.
pub struct GuardedEncoder {
    encoder: Option<wgpu::CommandEncoder>,
    active_encoders: Arc<std::sync::atomic::AtomicU32>,
}

impl std::ops::Deref for GuardedEncoder {
    type Target = wgpu::CommandEncoder;
    fn deref(&self) -> &Self::Target {
        self.encoder.as_ref().expect("encoder already finished")
    }
}

impl std::ops::DerefMut for GuardedEncoder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.encoder.as_mut().expect("encoder already finished")
    }
}

impl GuardedEncoder {
    /// Finish encoding and decrement the active encoder count.
    pub fn finish(mut self) -> wgpu::CommandBuffer {
        let cmd = self
            .encoder
            .take()
            .expect("encoder already finished")
            .finish();
        self.active_encoders.fetch_sub(1, Ordering::Release);
        cmd
    }
}

impl Drop for GuardedEncoder {
    fn drop(&mut self) {
        if self.encoder.is_some() {
            self.active_encoders.fetch_sub(1, Ordering::Release);
        }
    }
}

/// Wrapper around `Arc<wgpu::Device>` that auto-protects `create_*` calls.
///
/// Inherent methods shadow `wgpu::Device::create_*`, incrementing the
/// active-encoder counter before the call and decrementing after. This
/// prevents `device.poll()` from racing with resource creation on software
/// rasterizers (llvmpipe). Non-create methods pass through via `Deref`.
#[derive(Clone)]
pub struct GuardedDeviceHandle {
    inner: Arc<wgpu::Device>,
    active_encoders: Arc<AtomicU32>,
}

impl std::fmt::Debug for GuardedDeviceHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardedDeviceHandle")
            .field("inner", &"<wgpu::Device>")
            .finish()
    }
}

impl std::ops::Deref for GuardedDeviceHandle {
    type Target = wgpu::Device;
    fn deref(&self) -> &wgpu::Device {
        &self.inner
    }
}

impl GuardedDeviceHandle {
    pub(crate) fn new(inner: Arc<wgpu::Device>, active_encoders: Arc<AtomicU32>) -> Self {
        Self {
            inner,
            active_encoders,
        }
    }

    /// Get the underlying `Arc<wgpu::Device>` for ownership sharing.
    pub(crate) fn inner_arc(&self) -> Arc<wgpu::Device> {
        self.inner.clone()
    }

    fn guard(&self) {
        self.active_encoders.fetch_add(1, Ordering::Acquire);
    }

    fn unguard(&self) {
        self.active_encoders.fetch_sub(1, Ordering::Release);
    }

    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
        self.guard();
        let r = self.inner.create_buffer(desc);
        self.unguard();
        r
    }

    pub fn create_buffer_init(&self, desc: &wgpu::util::BufferInitDescriptor<'_>) -> wgpu::Buffer {
        self.guard();
        let r = self.inner.create_buffer_init(desc);
        self.unguard();
        r
    }

    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> wgpu::BindGroupLayout {
        self.guard();
        let r = self.inner.create_bind_group_layout(desc);
        self.unguard();
        r
    }

    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> wgpu::BindGroup {
        self.guard();
        let r = self.inner.create_bind_group(desc);
        self.unguard();
        r
    }

    pub fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> wgpu::PipelineLayout {
        self.guard();
        let r = self.inner.create_pipeline_layout(desc);
        self.unguard();
        r
    }

    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> wgpu::ComputePipeline {
        self.guard();
        let r = self.inner.create_compute_pipeline(desc);
        self.unguard();
        r
    }

    pub fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
    ) -> wgpu::ShaderModule {
        self.guard();
        let r = self.inner.create_shader_module(desc);
        self.unguard();
        r
    }

    pub fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> wgpu::CommandEncoder {
        self.guard();
        let r = self.inner.create_command_encoder(desc);
        self.unguard();
        r
    }
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
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    calibration: Option<GpuCalibration>,
    /// Vulkan pipeline cache — avoids re-compiling identical SPIR-V to machine code.
    pipeline_cache: Option<Arc<wgpu::PipelineCache>>,
    /// Set when the GPU reports a device-lost error.
    pub(crate) lost: Arc<AtomicBool>,
    /// Serializes submit+poll to prevent wgpu-core storage races.
    gpu_lock: Arc<std::sync::Mutex<()>>,
    /// Counts live GuardedEncoders — poll waits for this to reach zero.
    active_encoders: Arc<std::sync::atomic::AtomicU32>,
    /// Limits concurrent dispatches based on device capability.
    dispatch_semaphore: Arc<DispatchSemaphore>,
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
    pub fn name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get device type
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.adapter_info.device_type
    }

    /// Check if running on CPU fallback
    pub fn is_cpu(&self) -> bool {
        self.adapter_info.device_type == wgpu::DeviceType::Cpu
    }

    /// Check if the GPU driver has reported this device as lost.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Check if f64 shaders are truly available on this device.
    ///
    /// Returns `true` only when BOTH conditions hold:
    /// 1. `wgpu::Features::SHADER_F64` was granted at device creation.
    /// 2. The runtime probe (if completed) confirms basic f64 compilation works.
    ///
    /// groundSpring V37 discovered that NVK/NAK advertise `SHADER_F64` but fail
    /// to compile even basic f64 WGSL. This method consults the probe cache to
    /// avoid silent shader compilation failures.
    pub fn has_f64_shaders(&self) -> bool {
        if !self.device.features().contains(wgpu::Features::SHADER_F64) {
            return false;
        }
        match super::probe::cached_f64_builtins(self) {
            Some(caps) => caps.can_compile_f64(),
            None => true,
        }
    }

    /// Check if the Sovereign Compiler's SPIR-V passthrough path is available.
    ///
    /// Returns `true` when `wgpu::Features::SPIRV_SHADER_PASSTHROUGH` was
    /// granted at device creation — typically available on Vulkan backends
    /// (NVK, RADV, proprietary NVIDIA).
    pub fn has_spirv_passthrough(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
    }

    /// Access underlying wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get Arc-wrapped device (for shared ownership)
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        self.device.inner_arc()
    }

    /// Get the shared GPU lock (for components that call `device.poll` directly).
    pub(crate) fn gpu_lock_arc(&self) -> Arc<std::sync::Mutex<()>> {
        self.gpu_lock.clone()
    }

    /// Get the active encoder counter (for BufferPool poll coordination).
    pub(crate) fn active_encoders_arc(&self) -> Arc<std::sync::atomic::AtomicU32> {
        self.active_encoders.clone()
    }

    /// Increment active encoder count (encoding phase barrier).
    ///
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
    ///
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
    ///
    /// The returned `GuardedEncoder` increments the active encoder count,
    /// preventing `device.poll()` from running. Encoding is lock-free and
    /// never blocks other threads. Call `.finish()` to release.
    pub fn create_encoder_guarded(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> GuardedEncoder {
        self.active_encoders.fetch_add(1, Ordering::Acquire);
        let encoder = self.device.create_command_encoder(desc);
        GuardedEncoder {
            encoder: Some(encoder),
            active_encoders: self.active_encoders.clone(),
        }
    }

    /// Get adapter info (for capability detection)
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Get the device's pipeline cache for `create_compute_pipeline` calls.
    ///
    /// Returns `None` only when the Vulkan driver does not support pipeline caching.
    pub fn pipeline_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.pipeline_cache.as_deref()
    }

    /// Access command queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get Arc to command queue
    pub fn queue_arc(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }

    /// Acquire a dispatch permit from the device's concurrency budget.
    ///
    /// Hold this for the entire compile→bind→submit lifecycle of an
    /// operation. The device automatically sizes the budget to what the
    /// hardware can handle (2 for CPU/llvmpipe, 8 for discrete GPU, etc.).
    pub fn acquire_dispatch(&self) -> DispatchPermit<'_> {
        self.dispatch_semaphore.acquire()
    }

    /// Poll the device for completed work, catching device-lost panics.
    ///
    /// Every `device.poll(Maintain::Wait)` call site in barracuda should
    /// go through this method (or its `_nonblocking` variant) so that a
    /// driver-level device loss is consistently caught, flagged, and
    /// converted to `Err` instead of panicking the caller's thread.
    ///
    /// Serialized via `gpu_lock`: wgpu-core's internal resource tracking
    /// on software rasterizers can race under concurrent submit/poll.
    pub fn poll_safe(&self) -> Result<()> {
        if self.is_lost() {
            return Err(crate::error::BarracudaError::device_lost("device lost"));
        }
        let _guard = self.gpu_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.brief_encoder_wait();
        let device = self.device.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            device.poll(wgpu::Maintain::Wait);
        }));
        match result {
            Ok(()) => Ok(()),
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
    ///
    /// Serialized via `gpu_lock` to prevent wgpu-core storage races on
    /// software rasterizers. Use when the caller will poll separately
    /// (e.g. via `poll_safe()` or `map_staging_buffer()`).
    pub fn submit_commands(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        if self.is_lost() {
            return;
        }
        let _guard = self.gpu_lock.lock().unwrap_or_else(|e| e.into_inner());
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
    ///
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
            self.device.poll(wgpu::MaintainBase::Poll);
        }));
    }

    /// Extract a human-readable message from a `catch_unwind` payload.
    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
        payload
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown")
    }

    fn is_device_lost_panic(payload: &Box<dyn std::any::Any + Send>) -> bool {
        let msg = Self::panic_message(payload);
        msg.contains("lost") || msg.contains("Lost") || msg.contains("Parent device")
    }

    /// Submit commands and poll, respecting the device's concurrency budget.
    ///
    /// Acquires a dispatch permit (blocks if budget exhausted), then the
    /// GPU lock. Together they prevent both driver overload (semaphore) and
    /// state corruption (mutex).
    pub fn submit_and_poll(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        let _permit = self.dispatch_semaphore.acquire();
        self.submit_and_poll_inner(commands);
    }

    /// Submit without acquiring a dispatch permit.
    /// Use when the caller already holds a `DispatchPermit`.
    ///
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
        let _guard = self.gpu_lock.lock().unwrap_or_else(|e| e.into_inner());
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
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                device.poll(wgpu::Maintain::Wait);
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
    pub fn max_concurrent_dispatches(&self) -> usize {
        self.dispatch_semaphore.max_permits()
    }

    /// Execute WGSL compute shader
    ///
    /// Acquires a dispatch permit for the full compile→submit lifecycle.
    /// Holds encoding read-guard during compile/encode, releases before submit.
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
                    entry_point: "main",
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
                    pass.set_bind_group(i as u32, bg, &[]);
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
    pub fn optimal_workgroup_size(&self) -> u32 {
        self.calibration
            .as_ref()
            .map(|c| c.optimal_workgroup_size)
            .unwrap_or_else(|| GLOBAL_TUNER.get_or_calibrate(self).optimal_workgroup_size)
    }

    /// Get measured peak bandwidth for this device (GB/s)
    pub fn peak_bandwidth_gbps(&self) -> f64 {
        self.get_calibration().peak_bandwidth_gbps
    }

    /// Get measured dispatch overhead for this device (μs)
    pub fn dispatch_overhead_us(&self) -> f64 {
        self.get_calibration().dispatch_overhead_us
    }

    /// Create calibrated device (runs calibration immediately)
    pub async fn new_calibrated() -> Result<Self> {
        let mut device = Self::new().await?;
        let cal = device.get_calibration();
        device.calibration = Some(cal);
        Ok(device)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
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
        let adapters = WgpuDevice::enumerate_adapters();
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
        let adapters = WgpuDevice::enumerate_adapters();
        if adapters.is_empty() {
            return;
        }
        if let Ok(device) = WgpuDevice::with_adapter_selector("0").await {
            assert_eq!(device.name(), adapters[0].name);
        }
    }

    #[tokio::test]
    async fn test_adapter_selector_name_match() {
        let adapters = WgpuDevice::enumerate_adapters();
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
        let adapters = WgpuDevice::enumerate_adapters();
        let large_index = (adapters.len() + 1000).to_string();
        if let Err(e) = WgpuDevice::with_adapter_selector(&large_index).await {
            assert!(e.to_string().contains("No adapter matches"));
        }
    }

    #[tokio::test]
    async fn test_from_env_default() {
        std::env::remove_var(super::creation::ADAPTER_ENV_VAR);
        let _ = WgpuDevice::from_env().await;
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
