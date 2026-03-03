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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// WebGPU device - executes WGSL on any hardware
///
/// Concurrency is self-managed: `dispatch_semaphore` limits how many
/// operations can be in-flight simultaneously based on device type.
/// `gpu_lock` serializes the actual submit+poll to prevent wgpu state
/// corruption. Together they prevent both driver overload and data races.
#[derive(Debug, Clone)]
pub struct WgpuDevice {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    calibration: Option<GpuCalibration>,
    /// Vulkan pipeline cache — avoids re-compiling identical SPIR-V to machine code.
    pipeline_cache: Option<Arc<wgpu::PipelineCache>>,
    /// Set when the GPU reports a device-lost error.
    pub(crate) lost: Arc<AtomicBool>,
    /// Serializes GPU operations (submit, poll, map) across threads.
    gpu_lock: Arc<std::sync::Mutex<()>>,
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
        self.device.clone()
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

    /// Acquire the GPU operation lock. All code that touches `queue.submit`,
    /// `device.poll`, or `map_async` + `get_mapped_range` must hold this lock.
    /// Returns an RAII guard that releases on drop.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.gpu_lock.lock().unwrap_or_else(|e| e.into_inner())
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
    pub fn poll_safe(&self) -> Result<()> {
        if self.is_lost() {
            return Err(crate::error::BarracudaError::device_lost("device lost"));
        }
        let _guard = self.lock();
        self.poll_safe_unlocked()
    }

    /// Like `poll_safe` but does NOT acquire the gpu_lock.
    ///
    /// Use when the caller already holds the lock (e.g., inside
    /// `submit_and_poll_inner` after submit completes).
    fn poll_safe_unlocked(&self) -> Result<()> {
        let device = self.device.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            device.poll(wgpu::Maintain::Wait);
        }));
        match result {
            Ok(()) => Ok(()),
            Err(payload) => {
                self.lost.store(true, Ordering::Release);
                let msg = Self::panic_message(&payload);
                tracing::warn!("poll_safe: device lost: {msg}");
                Err(crate::error::BarracudaError::device(format!(
                    "GPU device lost during poll: {msg}"
                )))
            }
        }
    }

    /// Non-blocking poll variant for drain/housekeeping paths.
    pub fn poll_nonblocking(&self) {
        if self.is_lost() {
            return;
        }
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
    /// Device-lost panics from wgpu are caught and converted to a
    /// `lost` flag. Subsequent readback will see the lost state and
    /// return an error instead of panicking the test thread.
    pub(crate) fn submit_and_poll_inner(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        if self.is_lost() {
            return;
        }
        let _guard = self.lock();
        let queue = self.queue.clone();
        let device = self.device.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            queue.submit(commands);
            device.poll(wgpu::Maintain::Wait);
        }));
        if let Err(payload) = result {
            self.lost.store(true, Ordering::Release);
            let msg = Self::panic_message(&payload);
            if Self::is_device_lost_panic(&payload) {
                tracing::warn!("submit_and_poll: device lost (flagged, not panicking): {msg}");
                return;
            }
            std::panic::resume_unwind(payload);
        }
    }

    /// The device's concurrency budget (max simultaneous dispatches).
    pub fn max_concurrent_dispatches(&self) -> usize {
        self.dispatch_semaphore.max_permits()
    }

    /// Execute WGSL compute shader
    ///
    /// Acquires a dispatch permit for the full compile→submit lifecycle.
    pub fn execute_compute(
        &self,
        shader_source: &str,
        bind_groups: &[&wgpu::BindGroup],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let _permit = self.acquire_dispatch();
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

        self.submit_and_poll_inner(Some(encoder.finish()));
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

    #[cfg(feature = "toadstool")]
    #[tokio::test]
    async fn test_from_selection_gpu() {
        use super::super::toadstool_integration::DeviceSelection;
        if let Ok(device) = WgpuDevice::from_selection(DeviceSelection::Gpu).await {
            assert!(!device.is_cpu());
        }
    }

    #[cfg(feature = "toadstool")]
    #[tokio::test]
    async fn test_from_selection_cpu() {
        use super::super::toadstool_integration::DeviceSelection;
        if let Ok(device) = WgpuDevice::from_selection(DeviceSelection::Cpu).await {
            assert!(device.is_cpu());
        }
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
