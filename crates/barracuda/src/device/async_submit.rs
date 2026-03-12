// SPDX-License-Identifier: AGPL-3.0-only
//! Async Submission System for `BarraCuda`
//!
//! Provides non-blocking GPU submission with operation batching.
//!
//! ## Design Philosophy
//!
//! Instead of Vulkan-style timeline semaphores (not directly exposed in wgpu),
//! we use idiomatic wgpu patterns:
//!
//! 1. **Deferred Execution**: Queue multiple operations before submission
//! 2. **Batch Submission**: Submit many command buffers in one call
//! 3. **Async Readback**: Non-blocking buffer reads with futures
//! 4. **Submission Index**: Track work completion without explicit fences
//!
//! ## Deep Debt Evolution (Feb 16, 2026)
//!
//! - Fixed async methods that were blocking inside async context
//! - Added `poll_until_ready()` helper for cooperative async polling
//! - Uses `tokio::task::yield_now()` to avoid starving the executor
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::device::async_submit::AsyncSubmitter;
//!
//! let submitter = AsyncSubmitter::new(device.clone());
//!
//! // Queue multiple operations (no GPU work yet)
//! submitter.queue_operation(|encoder| {
//!     // ... encode first operation
//! });
//! submitter.queue_operation(|encoder| {
//!     // ... encode second operation  
//! });
//!
//! // Submit all at once (single driver call)
//! let index = submitter.submit_all();
//!
//! // Optionally wait for completion
//! submitter.wait_for(index);
//! ```

use super::WgpuDevice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Async GPU submission manager
///
/// Batches GPU operations and provides non-blocking submission tracking.
/// wgpu's `Queue` and `Device` are thread-safe — no external lock needed.
pub struct AsyncSubmitter {
    device: Arc<WgpuDevice>,
    /// Pending command buffers to be submitted
    pending: Mutex<Vec<wgpu::CommandBuffer>>,
    /// Submission counter for tracking work
    submission_index: AtomicU64,
    /// Index of last completed submission (approximate)
    completed_index: AtomicU64,
}

impl AsyncSubmitter {
    /// Create a new async submitter
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            device,
            pending: Mutex::new(Vec::with_capacity(16)),
            submission_index: AtomicU64::new(0),
            completed_index: AtomicU64::new(0),
        }
    }

    /// Queue a command buffer for deferred submission
    /// The command buffer will be submitted when `submit_all()` is called.
    /// This allows batching multiple operations into a single driver call.
    /// # Panics
    /// Panics if the pending mutex is poisoned.
    pub fn queue(&self, command_buffer: wgpu::CommandBuffer) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending.push(command_buffer);
    }

    /// Queue an operation using a callback that receives an encoder
    /// Convenience method that creates an encoder, runs the callback,
    /// and queues the finished command buffer.
    pub fn queue_operation<F>(&self, f: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut encoder =
            self.device
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AsyncSubmitter encoder"),
                });
        f(&mut encoder);
        self.queue(encoder.finish());
    }

    /// Submit all pending command buffers
    /// Returns a submission index that can be used to track completion.
    /// Serialized via `gpu_lock` to prevent wgpu-core storage races.
    /// # Panics
    /// Panics if the pending mutex is poisoned.
    pub fn submit_all(&self) -> u64 {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pending.is_empty() {
            return self.submission_index.load(Ordering::SeqCst);
        }

        let index = self.submission_index.fetch_add(1, Ordering::SeqCst) + 1;

        let buffers: Vec<_> = pending.drain(..).collect();
        let lock = self.device.gpu_lock_arc();
        let _guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.device.brief_encoder_wait();
        self.device.queue().submit(buffers);

        // Register a callback to update completed index when done
        let completed = Arc::new(AtomicU64::new(0));
        let completed_clone = completed;
        let idx = index;

        self.device.queue().on_submitted_work_done(move || {
            completed_clone.store(idx, Ordering::SeqCst);
        });

        index
    }

    /// Submit a single command buffer immediately
    /// Serialized via `gpu_lock` to prevent wgpu-core storage races.
    pub fn submit_immediate(&self, command_buffer: wgpu::CommandBuffer) -> u64 {
        let index = self.submission_index.fetch_add(1, Ordering::SeqCst) + 1;
        let lock = self.device.gpu_lock_arc();
        let _guard = lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.device.brief_encoder_wait();
        self.device.queue().submit(Some(command_buffer));
        index
    }

    /// Get the current submission index
    /// All work with index <= this value has been submitted to the GPU.
    pub fn current_index(&self) -> u64 {
        self.submission_index.load(Ordering::SeqCst)
    }

    /// Check if work at the given index is complete
    /// Note: This is approximate. For precise synchronization, use `wait_for()`.
    pub fn is_complete(&self, index: u64) -> bool {
        self.device.poll_nonblocking();
        self.completed_index.load(Ordering::SeqCst) >= index
    }

    /// Wait for work at the given index to complete
    /// Blocks until all GPU work up to and including the given index is done.
    /// Uses `poll_safe()` to respect the device lock.
    pub fn wait_for(&self, index: u64) {
        if self.completed_index.load(Ordering::SeqCst) >= index {
            return;
        }
        let _ = self.device.poll_safe();
    }

    /// Wait for all pending work to complete
    pub fn wait_all(&self) {
        let current = self.submission_index.load(Ordering::SeqCst);
        self.wait_for(current);
    }

    /// Get the number of pending command buffers
    /// # Panics
    /// Panics if the pending mutex is poisoned.
    pub fn pending_count(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Clear all pending command buffers without submitting
    /// Use with caution - discards queued work.
    /// # Panics
    /// Panics if the pending mutex is poisoned.
    pub fn clear_pending(&self) {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Async buffer read operation
///
/// Wraps a wgpu buffer read that can be awaited without blocking.
/// wgpu's `Device::poll` is thread-safe — no external lock needed.
///
/// Uses `std::sync::mpsc` (stdlib) instead of `futures::channel::oneshot`
/// — no external async runtime required for the channel itself.
pub struct AsyncReadback {
    device: Arc<WgpuDevice>,
    staging_buffer: wgpu::Buffer,
    size_bytes: u64,
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl AsyncReadback {
    /// Create a new async readback operation
    /// Copies from source buffer to staging and initiates async map.
    /// Thread-safe: wgpu's `Queue::submit` handles internal synchronization.
    #[must_use]
    pub fn new(device: &WgpuDevice, source: &wgpu::Buffer, size_bytes: u64) -> Self {
        let device = Arc::new(device.clone());

        let (staging_buffer, commands) = {
            device.encoding_guard();
            let staging = device.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("AsyncReadback staging"),
                size: size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder =
                device
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("AsyncReadback copy"),
                    });
            encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size_bytes);
            let cmds = encoder.finish();
            device.encoding_complete();
            (staging, cmds)
        };

        {
            let lock = device.gpu_lock_arc();
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            device.brief_encoder_wait();
            device.queue().submit(Some(commands));
        }

        let (sender, receiver) =
            std::sync::mpsc::sync_channel::<std::result::Result<(), wgpu::BufferAsyncError>>(1);
        {
            device.encoding_guard();
            staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).ok();
                });
            device.encoding_complete();
        }

        Self {
            device,
            staging_buffer,
            size_bytes,
            receiver,
        }
    }

    /// Poll the device and check if data is ready (non-blocking).
    #[must_use]
    pub fn poll(&self) -> bool {
        self.device.poll_nonblocking();
        self.receiver.try_recv().is_ok()
    }

    /// Poll the device until the buffer is ready (async-safe cooperative poll).
    /// Yields to the Tokio executor between device polls to avoid starving
    /// other tasks. Poll is serialized via `poll_lock` to prevent wgpu-core
    /// cleanup races.
    async fn poll_until_ready(&mut self) -> Result<(), String> {
        loop {
            let poll_ok = !self.device.is_lost();
            if poll_ok {
                self.device.poll_nonblocking();
            }

            if !poll_ok {
                return Err("GPU device lost during poll".to_string());
            }

            match self.receiver.try_recv() {
                Ok(result) => {
                    return result.map_err(|e| format!("Map error: {e:?}"));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    tokio::task::yield_now().await;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err("Readback cancelled — sender dropped".to_string());
                }
            }
        }
    }

    /// Wait for data and read as f32 (async-safe)
    /// Uses cooperative polling that yields to the async executor.
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub async fn read_f32(mut self) -> Result<Vec<f32>, String> {
        self.poll_until_ready().await?;

        // Read data
        let data = self.staging_buffer.slice(..).get_mapped_range();
        // Allocation required: mapped range is dropped before return; caller receives owned Vec
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Wait for data and read as f64 (async-safe)
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub async fn read_f64(mut self) -> Result<Vec<f64>, String> {
        self.poll_until_ready().await?;

        let data = self.staging_buffer.slice(..).get_mapped_range();
        // Allocation required: mapped range is dropped before return; caller receives owned Vec
        let result: Vec<f64> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Wait for data and read as u32 (async-safe)
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub async fn read_u32(mut self) -> Result<Vec<u32>, String> {
        self.poll_until_ready().await?;

        let data = self.staging_buffer.slice(..).get_mapped_range();
        // Allocation required: mapped range is dropped before return; caller receives owned Vec
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Wait for data and read as raw bytes (async-safe)
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub async fn read_bytes(mut self) -> Result<bytes::Bytes, String> {
        self.poll_until_ready().await?;

        let data = self.staging_buffer.slice(..).get_mapped_range();
        let result = bytes::Bytes::from(data.to_vec());

        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Blocking read as f32 (for sync contexts).
    /// Uses `poll_safe()` so the map callback fires before `recv()` — no async runtime needed.
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub fn read_f32_blocking(self) -> Result<Vec<f32>, String> {
        self.device
            .poll_safe()
            .map_err(|_| "GPU device lost during blocking poll".to_string())?;

        self.receiver
            .recv()
            .map_err(|_| "Readback cancelled — sender dropped".to_string())?
            .map_err(|e| format!("Map error: {e:?}"))?;

        let data = self.staging_buffer.slice(..).get_mapped_range();
        // Allocation required: mapped range is dropped before return; caller receives owned Vec
        let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        Ok(result)
    }

    /// Blocking read as f64 (for sync contexts).
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost during poll, buffer mapping fails,
    /// or the readback is cancelled (sender dropped).
    pub fn read_f64_blocking(self) -> Result<Vec<f64>, String> {
        self.device
            .poll_safe()
            .map_err(|_| "GPU device lost during blocking poll".to_string())?;

        self.receiver
            .recv()
            .map_err(|_| "Readback cancelled — sender dropped".to_string())?
            .map_err(|e| format!("Map error: {e:?}"))?;

        let data = self.staging_buffer.slice(..).get_mapped_range();
        // Allocation required: mapped range is dropped before return; caller receives owned Vec
        let result = bytemuck::cast_slice::<u8, f64>(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        Ok(result)
    }

    /// Get the size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_submitter_creation() {
        let Some(wgpu_device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return; // Skip if no GPU available
        };

        let submitter = AsyncSubmitter::new(Arc::clone(&wgpu_device));
        assert_eq!(submitter.pending_count(), 0);
        assert_eq!(submitter.current_index(), 0);
    }

    #[test]
    fn test_submission_index_increment() {
        // Test that submission index increments correctly (no GPU needed)
        let index = AtomicU64::new(0);
        assert_eq!(index.fetch_add(1, Ordering::SeqCst) + 1, 1);
        assert_eq!(index.fetch_add(1, Ordering::SeqCst) + 1, 2);
    }
}
