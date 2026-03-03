// SPDX-License-Identifier: AGPL-3.0-or-later
//! Internal buffer storage abstraction for `Tensor`.
//!
//! `TensorBuffer` encapsulates the two ownership modes a `Tensor`'s GPU
//! allocation can be in:
//!
//! | Variant   | Ownership                    | Drop behaviour                  |
//! |-----------|------------------------------|---------------------------------|
//! | `Owned`   | Exclusive `Arc<wgpu::Buffer>` | Buffer freed when refcount → 0 |
//! | `Pooled`  | Shared `Arc<PooledBuffer>`   | Returns to pool on last drop   |
//!
//! This is an internal detail; callers work with `Tensor` and never construct
//! `TensorBuffer` directly.

use crate::device::tensor_context::PooledBuffer;
use std::sync::Arc;

/// GPU buffer storage for a `Tensor` — either exclusively owned or pool-backed.
///
/// The `Clone` implementation is always a cheap `Arc` reference increment;
/// it never copies GPU memory.
pub(crate) enum TensorBuffer {
    /// Heap-allocated buffer freed when the last `Arc` reference drops.
    Owned(Arc<wgpu::Buffer>),
    /// Pool-managed buffer returned to its pool when the last reference drops.
    Pooled(Arc<PooledBuffer>),
}

impl TensorBuffer {
    /// Borrow the underlying `wgpu::Buffer`.
    pub fn as_ref(&self) -> &wgpu::Buffer {
        match self {
            TensorBuffer::Owned(buf) => buf.as_ref(),
            TensorBuffer::Pooled(buf) => buf.buffer(),
        }
    }

    /// Extract the `Arc<wgpu::Buffer>` if this is an `Owned` buffer.
    ///
    /// Returns `None` for pooled buffers.  Used by zero-copy paths that need
    /// shared `Arc` ownership (`GpuTensorStorage::from_tensor`).
    pub fn try_arc(&self) -> Option<Arc<wgpu::Buffer>> {
        match self {
            TensorBuffer::Owned(arc) => Some(Arc::clone(arc)),
            TensorBuffer::Pooled(_) => None,
        }
    }
}

impl Clone for TensorBuffer {
    /// Cheap clone: increments an `Arc` reference count, never copies GPU memory.
    fn clone(&self) -> Self {
        match self {
            TensorBuffer::Owned(buf) => TensorBuffer::Owned(Arc::clone(buf)),
            TensorBuffer::Pooled(buf) => TensorBuffer::Pooled(Arc::clone(buf)),
        }
    }
}
