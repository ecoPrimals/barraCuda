// SPDX-License-Identifier: AGPL-3.0-or-later
//! Buffer pool for zero-overhead tensor operations
//!
//! Buffers returned to the pool are held in a pending queue until a
//! non-blocking `device.poll(MaintainBase::Poll)` confirms that all
//! previously submitted GPU work has progressed. This prevents the
//! "drop-before-completion" race (S-13) where a recycled buffer could
//! be handed out while the GPU is still reading/writing its contents.

use crate::error::{BarracudaError, Result};
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

/// A buffer that automatically returns to its pool when dropped.
///
/// The buffer enters a pending queue on drop rather than being
/// immediately available for reuse — see [`BufferPool::drain_pending`].
pub struct PooledBuffer {
    buffer: Option<wgpu::Buffer>,
    pool: Weak<BufferPoolInner>,
    bucket: usize,
}

impl PooledBuffer {
    pub(crate) fn new(buffer: wgpu::Buffer, pool: Weak<BufferPoolInner>, bucket: usize) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
            bucket,
        }
    }

    /// Get the underlying wgpu buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("Buffer already taken")
    }

    /// Get the buffer size in bytes
    pub fn size(&self) -> u64 {
        self.buffer().size()
    }

    /// Convert to a regular wgpu::Buffer (removes from pool management)
    pub fn into_buffer(mut self) -> wgpu::Buffer {
        self.buffer.take().expect("Buffer already taken")
    }
}

impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        self.buffer()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Some(pool) = self.pool.upgrade() {
                pool.defer_return(buffer, self.bucket);
            }
        }
    }
}

/// Inner pool structure (separate to allow Weak references)
pub(crate) struct BufferPoolInner {
    pub pools: RwLock<HashMap<usize, Vec<wgpu::Buffer>>>,
    pub device: wgpu::Device,
    /// Serializes poll against submit (shared with WgpuDevice::gpu_lock).
    poll_lock: Arc<Mutex<()>>,
    /// Counts live GuardedEncoders — poll skips when > 0.
    active_encoders: Arc<AtomicU32>,
    pub allocations: AtomicUsize,
    pub reuses: AtomicUsize,
    pub solver_buffers: RwLock<HashMap<String, SolverBufferSet>>,
    pending: Mutex<Vec<(wgpu::Buffer, usize)>>,
}

impl BufferPoolInner {
    fn return_buffer(&self, buffer: wgpu::Buffer, bucket: usize) {
        self.pools
            .write()
            .expect("pools poisoned")
            .entry(bucket)
            .or_default()
            .push(buffer);
        self.reuses.fetch_add(1, Ordering::Relaxed);
    }

    /// Place a buffer into the pending queue instead of making it
    /// immediately available. The buffer becomes reusable only after
    /// [`BufferPool::drain_pending`] calls `device.poll`.
    fn defer_return(&self, buffer: wgpu::Buffer, bucket: usize) {
        self.pending
            .lock()
            .expect("pending lock poisoned")
            .push((buffer, bucket));
    }
}

/// Descriptor for creating a pinned solver buffer.
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    /// Buffer size in bytes.
    pub size: u64,
    /// wgpu buffer usage flags.
    pub usage: wgpu::BufferUsages,
    /// Optional debug label.
    pub label: Option<String>,
}

impl BufferDescriptor {
    /// Create a descriptor with default storage + copy usage.
    pub fn new(size: u64) -> Self {
        Self {
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    /// Descriptor for an f64 array of given count.
    pub fn f64_array(count: usize) -> Self {
        Self::new((count * std::mem::size_of::<f64>()) as u64)
    }

    /// Descriptor for an f32 array of given count.
    pub fn f32_array(count: usize) -> Self {
        Self::new((count * std::mem::size_of::<f32>()) as u64)
    }

    /// Set the buffer label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set buffer usage flags.
    pub fn with_usage(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage = usage;
        self
    }
}

/// A set of buffers pinned for the lifetime of a solver.
#[derive(Debug)]
pub struct SolverBufferSet {
    solver_id: String,
    buffers: HashMap<String, Arc<wgpu::Buffer>>,
}

impl SolverBufferSet {
    /// Get a buffer by name.
    pub fn get(&self, name: &str) -> Option<&wgpu::Buffer> {
        self.buffers.get(name).map(|b| b.as_ref())
    }

    /// Get a buffer as Arc by name.
    pub fn get_arc(&self, name: &str) -> Option<Arc<wgpu::Buffer>> {
        self.buffers.get(name).cloned()
    }

    /// Solver identifier.
    pub fn solver_id(&self) -> &str {
        &self.solver_id
    }

    /// Iterator over buffer names.
    pub fn buffer_names(&self) -> impl Iterator<Item = &str> {
        self.buffers.keys().map(|s| s.as_str())
    }

    /// Number of buffers in the set.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Returns true if the set has no buffers.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Memory pool for buffer reuse
pub struct BufferPool {
    inner: Arc<BufferPoolInner>,
}

impl BufferPool {
    /// Create a new buffer pool with shared locks.
    pub fn new(
        device: wgpu::Device,
        poll_lock: Arc<Mutex<()>>,
        active_encoders: Arc<AtomicU32>,
    ) -> Self {
        Self {
            inner: Arc::new(BufferPoolInner {
                pools: RwLock::new(HashMap::new()),
                device,
                poll_lock,
                active_encoders,
                allocations: AtomicUsize::new(0),
                reuses: AtomicUsize::new(0),
                solver_buffers: RwLock::new(HashMap::new()),
                pending: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Create a pool with its own independent locks (for tests/standalone use).
    pub fn new_standalone(device: wgpu::Device) -> Self {
        Self::new(
            device,
            Arc::new(Mutex::new(())),
            Arc::new(AtomicU32::new(0)),
        )
    }

    fn bucket_size(size: usize) -> usize {
        let min_size = 256;
        let size = size.max(min_size);
        size.next_power_of_two()
    }

    /// Try to move pending buffers back into the available pool.
    ///
    /// Uses `try_write()` on the poll_lock so it **never blocks encoding**.
    /// If encoding is in progress (read locks held), the drain is skipped
    /// entirely — buffers stay pending until the next drain attempt.
    ///
    /// Uses `MaintainBase::Poll` (non-blocking) to check for GPU completion
    /// without stalling. Buffers whose GPU work hasn't finished remain pending.
    pub fn drain_pending(&self) {
        let pending_count = self
            .inner
            .pending
            .lock()
            .expect("pending lock poisoned")
            .len();
        if pending_count == 0 {
            return;
        }
        if self.inner.active_encoders.load(Ordering::Acquire) > 0 {
            return;
        }
        let _guard = match self.inner.poll_lock.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::WouldBlock) => return,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.inner.device.poll(wgpu::PollType::Poll);
        }));

        let drained: Vec<(wgpu::Buffer, usize)> = {
            let mut pending = self.inner.pending.lock().expect("pending lock poisoned");
            std::mem::take(&mut *pending)
        };

        for (buffer, bucket) in drained {
            self.inner.return_buffer(buffer, bucket);
        }
    }

    /// Acquire a pooled buffer (returns to pool on drop).
    pub fn acquire_pooled(&self, size_bytes: usize) -> PooledBuffer {
        self.drain_pending();

        let bucket = Self::bucket_size(size_bytes);
        let buffer = self
            .inner
            .pools
            .write()
            .expect("pools poisoned")
            .get_mut(&bucket)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| self.allocate_new(bucket));
        PooledBuffer::new(buffer, Arc::downgrade(&self.inner), bucket)
    }

    /// Acquire a buffer (does not return to pool; call release explicitly).
    pub fn acquire(&self, size_bytes: usize) -> wgpu::Buffer {
        self.drain_pending();

        let bucket = Self::bucket_size(size_bytes);
        let recycled = self
            .inner
            .pools
            .write()
            .expect("pools poisoned")
            .get_mut(&bucket)
            .and_then(|v| v.pop());
        if let Some(buf) = recycled {
            self.inner.reuses.fetch_add(1, Ordering::Relaxed);
            return buf;
        }
        self.allocate_new(bucket)
    }

    fn allocate_new(&self, bucket: usize) -> wgpu::Buffer {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        self.inner.active_encoders.fetch_add(1, Ordering::Acquire);
        let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size: bucket as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.inner.active_encoders.fetch_sub(1, Ordering::Release);
        buf
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, buffer: wgpu::Buffer) {
        let bucket = Self::bucket_size(buffer.size() as usize);
        self.inner
            .pools
            .write()
            .expect("pools poisoned")
            .entry(bucket)
            .or_default()
            .push(buffer);
    }

    /// Returns (allocations, reuses).
    pub fn stats(&self) -> (usize, usize) {
        (
            self.inner.allocations.load(Ordering::Relaxed),
            self.inner.reuses.load(Ordering::Relaxed),
        )
    }

    /// Pin buffers for a solver; returns error if solver already has buffers.
    pub fn pin_solver_buffers(
        &self,
        solver_id: &str,
        buffers: &[(&str, BufferDescriptor)],
    ) -> Result<SolverBufferSet> {
        let mut solver_buffers = self
            .inner
            .solver_buffers
            .write()
            .expect("solver_buffers lock poisoned");

        if solver_buffers.contains_key(solver_id) {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Solver '{solver_id}' already has pinned buffers. Call release_solver_buffers() first."
                ),
            });
        }

        let mut buffer_map = HashMap::new();
        for (name, desc) in buffers {
            let label = desc
                .label
                .clone()
                .unwrap_or_else(|| format!("{solver_id}:{name}"));
            let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: desc.size,
                usage: desc.usage,
                mapped_at_creation: false,
            });
            self.inner.allocations.fetch_add(1, Ordering::Relaxed);
            buffer_map.insert(name.to_string(), Arc::new(buffer));
        }

        let buffer_set = SolverBufferSet {
            solver_id: solver_id.to_string(),
            buffers: buffer_map.clone(),
        };
        solver_buffers.insert(solver_id.to_string(), buffer_set);

        Ok(SolverBufferSet {
            solver_id: solver_id.to_string(),
            buffers: buffer_map,
        })
    }

    /// Release pinned buffers for a solver. Returns true if buffers were released.
    pub fn release_solver_buffers(&self, solver_id: &str) -> bool {
        let mut solver_buffers = self
            .inner
            .solver_buffers
            .write()
            .expect("solver_buffers lock poisoned");
        solver_buffers.remove(solver_id).is_some()
    }

    /// Get pinned buffers for a solver (cloned).
    pub fn get_solver_buffers(&self, solver_id: &str) -> Option<SolverBufferSet> {
        let solver_buffers = self
            .inner
            .solver_buffers
            .read()
            .expect("solver_buffers lock poisoned");
        solver_buffers.get(solver_id).map(|set| SolverBufferSet {
            solver_id: set.solver_id.clone(),
            buffers: set.buffers.clone(),
        })
    }

    /// Returns true if solver has pinned buffers.
    pub fn has_solver_buffers(&self, solver_id: &str) -> bool {
        let solver_buffers = self
            .inner
            .solver_buffers
            .read()
            .expect("solver_buffers lock poisoned");
        solver_buffers.contains_key(solver_id)
    }

    /// All solver IDs with pinned buffers.
    pub fn solver_ids(&self) -> Vec<String> {
        let solver_buffers = self
            .inner
            .solver_buffers
            .read()
            .expect("solver_buffers lock poisoned");
        solver_buffers.keys().cloned().collect()
    }
}
