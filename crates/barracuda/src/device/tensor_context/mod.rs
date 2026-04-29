// SPDX-License-Identifier: AGPL-3.0-or-later
//! `TensorContext` - Zero-overhead Tensor operations via internal pooling
//!
//! # Batched dispatch (neuralSpring handoff #4 / #10)
//!
//! Each Tensor op that routes through [`TensorContext::record_operation`] is
//! added to a pending queue when batching mode is active.  A single
//! `CommandEncoder` is flushed on [`TensorContext::end_batch`], collapsing N
//! `queue.submit()` calls into one.
//!
//! The ergonomic entry point is [`TensorSession`]:
//!
//! ```ignore
//! use barracuda::device::tensor_context::{get_device_context, BatchGuard};
//!
//! let guard = BatchGuard::new(&device);
//! let c = a.add(&b)?;   // queued, not yet submitted
//! let d = c.add(&e)?;   // queued
//! guard.flush()?;        // single queue.submit() covering both ops
//! ```
//!
//! **Note**: Only ops wired through `record_operation()` participate in
//! batching.  The wiring is incremental; `add` is the first op to support it.

mod context;
mod limits;
mod pool;

pub use context::{TensorContext, TensorContextStats, clear_global_contexts, get_device_context};
pub use limits::{high_capacity_limits, science_limits};
pub use pool::{BufferDescriptor, BufferPool, PooledBuffer, SolverBufferSet};

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// RAII guard for batched tensor dispatch.
///
/// Calls [`TensorContext::begin_batch`] on creation and
/// [`TensorContext::end_batch`] (flushing all pending ops) on
/// [`BatchGuard::flush`] or [`Drop`].
///
/// This is the low-level batch primitive. For a full op-batching session
/// with recorded operations (`add`, `matmul`, `relu`, etc.), use
/// [`crate::session::TensorSession`] instead.
///
/// # Example
/// ```ignore
/// let guard = BatchGuard::new(&device);
/// let sum = a.add(&b)?;
/// let out = sum.add(&c)?;
/// guard.flush()?;  // one queue.submit() instead of two
/// ```
pub struct BatchGuard {
    ctx: Arc<TensorContext>,
}

/// Legacy alias for [`BatchGuard`].
#[deprecated(
    since = "0.3.12",
    note = "renamed to BatchGuard to avoid collision with session::TensorSession"
)]
pub type TensorSession = BatchGuard;

impl BatchGuard {
    /// Begin a batch session on `device`.
    #[must_use]
    pub fn new(device: &Arc<WgpuDevice>) -> Self {
        let ctx = get_device_context(device);
        ctx.begin_batch();
        Self { ctx }
    }

    /// Submit all queued ops in a single `queue.submit()` and end the session.
    /// # Errors
    /// Returns [`Err`] if the GPU device is lost or command submission fails.
    pub fn flush(self) -> Result<()> {
        self.ctx.end_batch()
    }
}

impl Drop for BatchGuard {
    /// Flush on drop so batched ops are not silently lost if `flush()` is
    /// not called explicitly.
    fn drop(&mut self) {
        if self.ctx.is_batching() {
            if let Err(e) = self.ctx.end_batch() {
                tracing::debug!("BatchGuard::drop end_batch failed: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device;

    async fn create_test_device() -> wgpu::Device {
        get_test_device().await.device_clone()
    }

    #[tokio::test]
    async fn test_buffer_pool_basic_acquire_release() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        let buf1 = pool.acquire(1024);
        assert!(buf1.size() >= 1024);
        let size = buf1.size();
        pool.release(buf1);
        let buf2 = pool.acquire(1024);
        assert_eq!(buf2.size(), size);
        let (allocs, reuses) = pool.stats();
        assert_eq!(allocs, 1);
        assert_eq!(reuses, 1);
    }

    #[tokio::test]
    async fn test_buffer_pool_power_of_two_bucketing() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        assert_eq!(pool.acquire(1000).size(), 1024);
        assert_eq!(pool.acquire(1025).size(), 2048);
        assert_eq!(pool.acquire(100).size(), 256);
    }

    #[tokio::test]
    async fn test_buffer_pool_multiple_buckets() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        let buf_256 = pool.acquire(256);
        let buf_1024 = pool.acquire(1024);
        let buf_4096 = pool.acquire(4096);
        pool.release(buf_256);
        pool.release(buf_1024);
        pool.release(buf_4096);
        assert_eq!(pool.acquire(200).size(), 256);
        assert_eq!(pool.acquire(1000).size(), 1024);
        assert_eq!(pool.acquire(4000).size(), 4096);
    }

    #[tokio::test]
    async fn test_pooled_buffer_auto_return() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        let (allocs_before, reuses_before) = pool.stats();
        {
            let _ = pool.acquire_pooled(1024);
        }
        let _ = pool.acquire_pooled(1024);
        let (allocs_after, reuses_after) = pool.stats();
        assert_eq!(allocs_after - allocs_before, 1);
        assert_eq!(reuses_after - reuses_before, 1);
    }

    #[tokio::test]
    async fn test_tensor_context_acquire_pooled() {
        let device = get_test_device().await;
        let ctx = TensorContext::new(device);
        assert!(ctx.acquire_pooled_output(1000).size() >= 4000);
    }

    #[tokio::test]
    async fn test_tensor_context_batching_mode() {
        let device = get_test_device().await;
        let ctx = TensorContext::new(device);
        assert!(!ctx.is_batching());
        ctx.begin_batch();
        assert!(ctx.is_batching());
        ctx.end_batch().expect("end_batch failed");
        assert!(!ctx.is_batching());
    }

    #[tokio::test]
    async fn test_tensor_context_stats() {
        let device = get_test_device().await;
        let ctx = TensorContext::new(device);
        let stats = ctx.stats();
        assert_eq!(stats.buffer_allocations, 0);
        assert_eq!(stats.ops_executed, 0);
    }

    #[tokio::test]
    async fn test_global_context_registry() {
        let device = get_test_device().await;
        let ctx1 = get_device_context(&device);
        let ctx2 = get_device_context(&device);
        let initial = ctx1.stats().buffer_allocations;
        let _ = ctx1.acquire_pooled_output(1000);
        assert!(ctx2.stats().buffer_allocations >= initial);
    }

    #[test]
    fn test_high_capacity_limits() {
        let limits = high_capacity_limits();
        assert_eq!(limits.max_storage_buffer_binding_size, 1 << 30);
        assert_eq!(limits.max_buffer_size, 1 << 31);
    }

    #[tokio::test]
    async fn test_pin_solver_buffers_basic() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        let buffers = pool
            .pin_solver_buffers(
                "test_solver",
                &[
                    ("matrix_a", BufferDescriptor::f64_array(100)),
                    ("result", BufferDescriptor::f64_array(100)),
                ],
            )
            .expect("pin failed");
        assert_eq!(buffers.len(), 2);
        assert!(buffers.get("matrix_a").is_some());
        assert!(pool.release_solver_buffers("test_solver"));
    }

    #[tokio::test]
    async fn test_pin_solver_buffers_duplicate_id_error() {
        let device = create_test_device().await;
        let pool = BufferPool::new_standalone(device);
        pool.pin_solver_buffers("solver_a", &[("buf", BufferDescriptor::new(256))])
            .expect("first pin");
        assert!(
            pool.pin_solver_buffers("solver_a", &[("buf2", BufferDescriptor::new(256))])
                .is_err()
        );
        let _ = pool.release_solver_buffers("solver_a");
        assert!(
            pool.pin_solver_buffers("solver_a", &[("buf2", BufferDescriptor::new(256))])
                .is_ok()
        );
    }

    #[tokio::test]
    async fn test_buffer_descriptor_helpers() {
        assert_eq!(BufferDescriptor::f64_array(100).size, 800);
        assert_eq!(BufferDescriptor::f32_array(100).size, 400);
        assert_eq!(
            BufferDescriptor::new(1024).with_label("x").label,
            Some("x".to_string())
        );
    }

    #[tokio::test]
    async fn test_tensor_context_solver_buffers() {
        let device = get_test_device().await;
        let ctx = TensorContext::new(device);
        let buffers = ctx
            .pin_solver_buffers(
                "scf",
                &[
                    ("hamiltonian", BufferDescriptor::f64_array(64)),
                    ("density", BufferDescriptor::f64_array(32)),
                ],
            )
            .expect("pin");
        assert_eq!(buffers.len(), 2);
        assert!(ctx.get_solver_buffers("scf").is_some());
        assert!(ctx.release_solver_buffers("scf"));
        assert!(ctx.get_solver_buffers("scf").is_none());
    }
}
