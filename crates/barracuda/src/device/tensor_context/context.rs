// SPDX-License-Identifier: AGPL-3.0-or-later
//! TensorContext - accelerated tensor operations via internal pooling

use super::pool::{BufferDescriptor, BufferPool, SolverBufferSet};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, DeviceFingerprint, GLOBAL_CACHE};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

type PendingOp = Box<dyn FnOnce(&mut wgpu::CommandEncoder) + Send>;

static GLOBAL_CONTEXTS: LazyLock<RwLock<HashMap<DeviceFingerprint, Arc<TensorContext>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[derive(Clone, PartialEq, Eq)]
struct BindGroupKey {
    layout_sig: BindGroupLayoutSignature,
    buffer_hashes: Vec<u64>,
}

impl std::hash::Hash for BindGroupKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.layout_sig.hash(state);
        self.buffer_hashes.hash(state);
    }
}

fn buffer_identity(buf: &wgpu::Buffer) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    buf.hash(&mut h);
    h.finish()
}

/// TensorContext - accelerated tensor operations via internal pooling
pub struct TensorContext {
    device: Arc<WgpuDevice>,
    buffer_pool: BufferPool,
    bind_group_cache: RwLock<HashMap<BindGroupKey, Arc<wgpu::BindGroup>>>,
    pending_ops: Mutex<Vec<PendingOp>>,
    batching: AtomicBool,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    ops_executed: AtomicUsize,
    ops_batched: AtomicUsize,
}

impl TensorContext {
    /// Create a new tensor context for the device.
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            buffer_pool: BufferPool::new(
                device.device_clone(),
                device.gpu_lock_arc(),
                device.active_encoders_arc(),
            ),
            device,
            bind_group_cache: RwLock::new(HashMap::new()),
            pending_ops: Mutex::new(Vec::new()),
            batching: AtomicBool::new(false),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            ops_executed: AtomicUsize::new(0),
            ops_batched: AtomicUsize::new(0),
        }
    }

    /// Start batching operations (defer submit until end_batch).
    pub fn begin_batch(&self) {
        self.batching.store(true, Ordering::SeqCst);
    }

    /// End batching and submit all pending operations.
    pub fn end_batch(&self) -> Result<()> {
        self.batching.store(false, Ordering::SeqCst);
        self.sync()
    }

    /// Returns true if batching is active.
    pub fn is_batching(&self) -> bool {
        self.batching.load(Ordering::SeqCst)
    }

    /// Acquire an output buffer (f32 elements). Does not return to pool.
    pub fn acquire_output_buffer(&self, size_elements: usize) -> wgpu::Buffer {
        self.buffer_pool
            .acquire(size_elements * std::mem::size_of::<f32>())
    }

    /// Acquire a pooled output buffer (returns to pool on drop).
    pub fn acquire_pooled_output(&self, size_elements: usize) -> super::pool::PooledBuffer {
        self.buffer_pool
            .acquire_pooled(size_elements * std::mem::size_of::<f32>())
    }

    /// Acquire a pooled output buffer sized for f64 elements.
    pub fn acquire_pooled_output_f64(&self, size_elements: usize) -> super::pool::PooledBuffer {
        self.buffer_pool
            .acquire_pooled(size_elements * std::mem::size_of::<f64>())
    }

    /// Acquire a pooled output buffer with explicit byte size.
    pub fn acquire_pooled_bytes(&self, size_bytes: usize) -> super::pool::PooledBuffer {
        self.buffer_pool.acquire_pooled(size_bytes)
    }

    /// Record an operation (immediate or batched).
    pub fn record_operation<F>(&self, op: F) -> Result<()>
    where
        F: FnOnce(&mut wgpu::CommandEncoder) + Send + 'static,
    {
        if self.is_batching() {
            self.pending_ops
                .lock()
                .expect("pending_ops poisoned")
                .push(Box::new(op));
            self.ops_batched.fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            let commands = {
                self.device.encoding_guard();
                let mut encoder =
                    self.device
                        .device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("TensorContext Immediate"),
                        });
                op(&mut encoder);
                let cmds = encoder.finish();
                self.device.encoding_complete();
                cmds
            };
            self.device.submit_commands(Some(commands));
            self.ops_executed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Get or create a bind group from layout signature and buffers.
    pub fn get_or_create_bind_group(
        &self,
        layout_sig: BindGroupLayoutSignature,
        buffers: &[&wgpu::Buffer],
        label: Option<&str>,
    ) -> Arc<wgpu::BindGroup> {
        let buffer_hashes: Vec<_> = buffers.iter().map(|b| buffer_identity(b)).collect();
        let key = BindGroupKey {
            layout_sig,
            buffer_hashes,
        };

        // Fast path — read lock (the common case: already cached).
        if let Some(bg) = self
            .bind_group_cache
            .read()
            .expect("bind_group_cache poisoned")
            .get(&key)
        {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return bg.clone();
        }

        // Slow path — create and insert.
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let layout = GLOBAL_CACHE.get_or_create_layout(
            self.device.device(),
            self.device.adapter_info(),
            layout_sig,
            label,
        );

        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = Arc::new(self.device.device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label,
                layout: &layout,
                entries: &entries,
            },
        ));
        // Use `entry().or_insert()` so a concurrent insert wins gracefully.
        self.bind_group_cache
            .write()
            .expect("bind_group_cache poisoned")
            .entry(key)
            .or_insert(bind_group)
            .clone()
    }

    /// Acquire a buffer (f32 elements).
    pub fn acquire_buffer(&self, size_elements: usize) -> wgpu::Buffer {
        self.buffer_pool
            .acquire(size_elements * std::mem::size_of::<f32>())
    }

    /// Return a buffer to the pool for reuse.
    pub fn release_buffer(&self, buffer: wgpu::Buffer) {
        self.buffer_pool.release(buffer);
    }

    /// Submit all pending batched operations.
    pub fn sync(&self) -> Result<()> {
        let mut pending = self.pending_ops.lock().expect("pending_ops poisoned");
        if pending.is_empty() {
            return Ok(());
        }

        let commands = {
            self.device.encoding_guard();
            let mut encoder =
                self.device
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("TensorContext Batch Encoder"),
                    });
            for op in pending.drain(..) {
                op(&mut encoder);
            }
            let cmds = encoder.finish();
            self.device.encoding_complete();
            cmds
        };
        self.device.submit_commands(Some(commands));
        Ok(())
    }

    /// The underlying device.
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// The buffer pool.
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Pin buffers for a solver (held until release_solver_buffers).
    pub fn pin_solver_buffers(
        &self,
        solver_id: &str,
        buffers: &[(&str, BufferDescriptor)],
    ) -> Result<SolverBufferSet> {
        self.buffer_pool.pin_solver_buffers(solver_id, buffers)
    }

    /// Release pinned buffers for a solver.
    pub fn release_solver_buffers(&self, solver_id: &str) -> bool {
        self.buffer_pool.release_solver_buffers(solver_id)
    }

    /// Get pinned buffers for a solver.
    pub fn get_solver_buffers(&self, solver_id: &str) -> Option<SolverBufferSet> {
        self.buffer_pool.get_solver_buffers(solver_id)
    }

    /// Get context statistics.
    pub fn stats(&self) -> TensorContextStats {
        let (allocs, reuses) = self.buffer_pool.stats();
        TensorContextStats {
            buffer_allocations: allocs,
            buffer_reuses: reuses,
            bind_group_cache_hits: self.cache_hits.load(Ordering::Relaxed),
            bind_group_cache_misses: self.cache_misses.load(Ordering::Relaxed),
            ops_executed: self.ops_executed.load(Ordering::Relaxed),
            ops_batched: self.ops_batched.load(Ordering::Relaxed),
        }
    }
}

/// TensorContext statistics.
#[derive(Debug, Clone)]
pub struct TensorContextStats {
    /// Number of new buffer allocations.
    pub buffer_allocations: usize,
    /// Number of buffer reuses from pool.
    pub buffer_reuses: usize,
    /// Bind group cache hits.
    pub bind_group_cache_hits: usize,
    /// Bind group cache misses.
    pub bind_group_cache_misses: usize,
    /// Operations executed immediately.
    pub ops_executed: usize,
    /// Operations batched (deferred).
    pub ops_batched: usize,
}

impl std::fmt::Display for TensorContextStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let buffer_hit_rate = if self.buffer_allocations + self.buffer_reuses > 0 {
            self.buffer_reuses as f64 / (self.buffer_allocations + self.buffer_reuses) as f64
                * 100.0
        } else {
            0.0
        };
        let bg_hit_rate = if self.bind_group_cache_hits + self.bind_group_cache_misses > 0 {
            self.bind_group_cache_hits as f64
                / (self.bind_group_cache_hits + self.bind_group_cache_misses) as f64
                * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "Buffers: {} allocs, {} reuses ({:.1}% reuse)\n\
             BindGroups: {} hits, {} misses ({:.1}% hit rate)\n\
             Operations: {} executed, {} batched",
            self.buffer_allocations,
            self.buffer_reuses,
            buffer_hit_rate,
            self.bind_group_cache_hits,
            self.bind_group_cache_misses,
            bg_hit_rate,
            self.ops_executed,
            self.ops_batched
        )
    }
}

/// Get or create the global TensorContext for a device.
///
/// Uses double-checked locking: fast read path for the common case,
/// write path only when a new device is first seen.
pub fn get_device_context(device: &Arc<WgpuDevice>) -> Arc<TensorContext> {
    let fingerprint = DeviceFingerprint::from_adapter_info(device.adapter_info());

    // Fast path.
    if let Some(ctx) = GLOBAL_CONTEXTS
        .read()
        .expect("GLOBAL_CONTEXTS poisoned")
        .get(&fingerprint)
    {
        return ctx.clone();
    }

    // Slow path — first time this device is seen.
    GLOBAL_CONTEXTS
        .write()
        .expect("GLOBAL_CONTEXTS poisoned")
        .entry(fingerprint)
        .or_insert_with(|| Arc::new(TensorContext::new(device.clone())))
        .clone()
}

/// Clear all global contexts (for testing/benchmarking).
pub fn clear_global_contexts() {
    GLOBAL_CONTEXTS
        .write()
        .expect("GLOBAL_CONTEXTS poisoned")
        .clear();
}
