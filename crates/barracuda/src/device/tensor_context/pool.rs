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
use std::sync::atomic::{AtomicUsize, Ordering};
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
    pub device: Arc<wgpu::Device>,
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

/// Descriptor for creating a pinned solver buffer
#[derive(Debug, Clone)]
pub struct BufferDescriptor {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub label: Option<String>,
}

impl BufferDescriptor {
    pub fn new(size: u64) -> Self {
        Self {
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label: None,
        }
    }

    pub fn f64_array(count: usize) -> Self {
        Self::new((count * std::mem::size_of::<f64>()) as u64)
    }

    pub fn f32_array(count: usize) -> Self {
        Self::new((count * std::mem::size_of::<f32>()) as u64)
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_usage(mut self, usage: wgpu::BufferUsages) -> Self {
        self.usage = usage;
        self
    }
}

/// A set of buffers pinned for the lifetime of a solver
#[derive(Debug)]
pub struct SolverBufferSet {
    solver_id: String,
    buffers: HashMap<String, Arc<wgpu::Buffer>>,
}

impl SolverBufferSet {
    pub fn get(&self, name: &str) -> Option<&wgpu::Buffer> {
        self.buffers.get(name).map(|b| b.as_ref())
    }

    pub fn get_arc(&self, name: &str) -> Option<Arc<wgpu::Buffer>> {
        self.buffers.get(name).cloned()
    }

    pub fn solver_id(&self) -> &str {
        &self.solver_id
    }

    pub fn buffer_names(&self) -> impl Iterator<Item = &str> {
        self.buffers.keys().map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Memory pool for buffer reuse
pub struct BufferPool {
    inner: Arc<BufferPoolInner>,
}

impl BufferPool {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            inner: Arc::new(BufferPoolInner {
                pools: RwLock::new(HashMap::new()),
                device,
                allocations: AtomicUsize::new(0),
                reuses: AtomicUsize::new(0),
                solver_buffers: RwLock::new(HashMap::new()),
                pending: Mutex::new(Vec::new()),
            }),
        }
    }

    fn bucket_size(size: usize) -> usize {
        let min_size = 256;
        let size = size.max(min_size);
        size.next_power_of_two()
    }

    /// Move pending buffers back into the available pool after a
    /// non-blocking device poll confirms completed GPU work.
    /// Called automatically before every `acquire` / `acquire_pooled`.
    pub fn drain_pending(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.device.poll(wgpu::MaintainBase::Poll);
        }));

        let drained: Vec<(wgpu::Buffer, usize)> = {
            let mut pending = self.inner.pending.lock().expect("pending lock poisoned");
            std::mem::take(&mut *pending)
        };

        for (buffer, bucket) in drained {
            self.inner.return_buffer(buffer, bucket);
        }
    }

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
        self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size: bucket as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

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

    pub fn stats(&self) -> (usize, usize) {
        (
            self.inner.allocations.load(Ordering::Relaxed),
            self.inner.reuses.load(Ordering::Relaxed),
        )
    }

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

    pub fn release_solver_buffers(&self, solver_id: &str) -> bool {
        let mut solver_buffers = self
            .inner
            .solver_buffers
            .write()
            .expect("solver_buffers lock poisoned");
        solver_buffers.remove(solver_id).is_some()
    }

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

    pub fn has_solver_buffers(&self, solver_id: &str) -> bool {
        let solver_buffers = self
            .inner
            .solver_buffers
            .read()
            .expect("solver_buffers lock poisoned");
        solver_buffers.contains_key(solver_id)
    }

    pub fn solver_ids(&self) -> Vec<String> {
        let solver_buffers = self
            .inner
            .solver_buffers
            .read()
            .expect("solver_buffers lock poisoned");
        solver_buffers.keys().cloned().collect()
    }
}
