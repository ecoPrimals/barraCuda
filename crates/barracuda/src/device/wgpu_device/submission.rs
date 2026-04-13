// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU submission pipeline — submit, poll, and panic recovery.
//!
//! All `device.poll()` and `queue.submit()` calls go through these methods
//! for consistent gpu-lock serialization and device-lost panic handling.
//! Extracted from `mod.rs` to keep the device facade under 600 lines.

use std::sync::atomic::Ordering;

use super::{WgpuDevice, poll_timeout};
use crate::error::Result;

impl WgpuDevice {
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
            self.handle_device_lost_panic(payload, "submit_commands");
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

    fn handle_device_lost_panic(&self, payload: Box<dyn std::any::Any + Send>, context: &str) {
        if Self::is_device_lost_panic(&payload) {
            self.lost.store(true, Ordering::Release);
            tracing::warn!("{context}: device lost: {}", Self::panic_message(&payload));
            return;
        }
        std::panic::resume_unwind(payload);
    }

    /// Submit without acquiring a dispatch permit.
    /// Use when the caller already holds a `DispatchPermit`.
    ///
    /// Submit and poll use **separate** lock acquisitions so other threads
    /// can interleave their submits while we poll.  `poll(Wait)` processes
    /// ALL pending work, so interleaved submits from other threads are
    /// completed too — no work is lost.
    pub(crate) fn submit_and_poll_inner(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        self.submit_commands_inner(commands);
        self.poll_wait_inner();
    }

    fn submit_commands_inner(&self, commands: impl IntoIterator<Item = wgpu::CommandBuffer>) {
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
            self.handle_device_lost_panic(payload, "submit_commands_inner");
        }
    }

    fn poll_wait_inner(&self) {
        if self.is_lost() {
            return;
        }
        let _guard = self
            .gpu_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.brief_encoder_wait();
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
                    "poll_wait_inner: GPU poll timed out after {}s",
                    timeout.map_or(0, |d| d.as_secs())
                );
            }
        }));
        if let Err(payload) = result {
            self.handle_device_lost_panic(payload, "poll_wait_inner");
        }
    }
}
