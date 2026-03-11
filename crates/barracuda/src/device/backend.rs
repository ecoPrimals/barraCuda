// SPDX-License-Identifier: AGPL-3.0-only
//! Backend-agnostic GPU compute trait.
//!
//! `GpuBackend` defines the minimal surface a compute backend must implement:
//! buffer lifecycle, shader compilation, and compute dispatch. `WgpuDevice`
//! implements it today; `CoralReefDevice` will implement it when `coral-gpu`
//! is available as a standalone crate, enabling sovereign dispatch without
//! the Vulkan stack.
//!
//! Ops that use `ComputeDispatch` work through this trait automatically.
//! Ops that use raw `wgpu::Device` calls can be migrated incrementally.

use crate::error::Result;
use std::sync::Arc;

/// Descriptor for a single buffer binding in a compute dispatch.
pub struct BufferBinding<'a, B: GpuBackend + ?Sized> {
    /// Bind group index (0, 1, 2, ...).
    pub index: u32,
    /// Reference to the backend-specific buffer.
    pub buffer: &'a B::Buffer,
    /// Whether the shader only reads this buffer.
    pub read_only: bool,
    /// Whether this is a uniform buffer (vs storage).
    pub is_uniform: bool,
}

/// Complete description of a compute dispatch operation.
///
/// Backends receive this from `ComputeDispatch::submit()` and execute the
/// entire compile → bind → dispatch → submit → sync lifecycle.
pub struct DispatchDescriptor<'a, B: GpuBackend + ?Sized> {
    /// Human-readable label for debug/profiling.
    pub label: &'a str,
    /// WGSL shader source code.
    pub shader_source: &'a str,
    /// Shader entry point name.
    pub entry_point: &'a str,
    /// Buffer bindings for the dispatch.
    pub bindings: Vec<BufferBinding<'a, B>>,
    /// Workgroup counts (x, y, z).
    pub workgroups: (u32, u32, u32),
    /// Use f64 shader compilation path.
    pub f64_shader: bool,
    /// Use DF64 (double-float f32-pair) compilation path.
    pub df64_shader: bool,
}

/// Backend-agnostic GPU compute interface.
///
/// The trait is intentionally minimal: 9 required methods covering identity,
/// buffer lifecycle, and compute dispatch. Typed convenience methods (f32/f64/u32
/// buffers, reads, writes) are provided as defaults using bytemuck.
///
/// # Associated Types
///
/// - `Buffer`: The backend's buffer handle. For wgpu this is `wgpu::Buffer`;
///   for coral-gpu it will be `coral_gpu::BufferHandle`.
///
/// # Design
///
/// The trait captures the **compute dispatch contract** — the part that
/// `ComputeDispatch` and most ops need. Backend-specific methods (raw
/// `wgpu::Device` access, pipeline caching, encoder guards) remain on
/// the concrete types for ops that need them during incremental migration.
pub trait GpuBackend: Send + Sync {
    /// Backend-specific buffer handle.
    type Buffer: Send + Sync;

    // ── Identity ──────────────────────────────────────────────────────

    /// Human-readable device name (e.g. "NVIDIA `GeForce` RTX 3080").
    fn name(&self) -> &str;

    /// Whether this device supports native f64 shaders.
    fn has_f64_shaders(&self) -> bool;

    /// Whether the device has been reported as lost by the driver.
    fn is_lost(&self) -> bool;

    // ── Buffer lifecycle ──────────────────────────────────────────────

    /// Allocate an empty storage buffer (read-write, copy src+dst).
    ///
    /// # Errors
    /// Returns [`Err`] if allocation fails (e.g. device lost, OOM).
    fn alloc_buffer(&self, label: &str, size: u64) -> Result<Self::Buffer>;

    /// Allocate a storage buffer initialized with `contents`.
    ///
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn alloc_buffer_init(&self, label: &str, contents: &[u8]) -> Result<Self::Buffer>;

    /// Allocate a uniform buffer initialized with `contents`.
    ///
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn alloc_uniform(&self, label: &str, contents: &[u8]) -> Result<Self::Buffer>;

    /// Write raw bytes to a buffer at `offset`.
    fn upload(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]);

    /// Read `size` bytes from a buffer back to host memory.
    ///
    /// Returns [`bytes::Bytes`] for zero-copy downstream consumers.
    ///
    /// # Errors
    /// Returns [`Err`] if readback fails (e.g. device lost, mapping error).
    fn download(&self, buffer: &Self::Buffer, size: u64) -> Result<bytes::Bytes>;

    // ── Compute dispatch ──────────────────────────────────────────────

    /// Execute a complete compute dispatch: compile shader, create bindings,
    /// record compute pass, submit, and wait for completion.
    ///
    /// This is the core abstraction — each backend implements the full
    /// pipeline using its own API (wgpu, coral-gpu, etc.).
    ///
    /// # Errors
    /// Returns [`Err`] if compilation, dispatch, or synchronization fails.
    fn dispatch_compute(&self, desc: DispatchDescriptor<'_, Self>) -> Result<()>;

    /// Execute multiple compute dispatches in a single GPU submission.
    ///
    /// On backends that support command batching (e.g. wgpu/Vulkan), all
    /// dispatches are recorded into one command encoder, amortizing the
    /// per-submission overhead (~1.6ms on Vulkan → ~0.1ms amortized).
    ///
    /// The default implementation calls [`dispatch_compute`] sequentially.
    /// Backends that support batching should override this.
    ///
    /// # Errors
    /// Returns [`Err`] if any dispatch fails.
    fn dispatch_compute_batch(&self, descs: Vec<DispatchDescriptor<'_, Self>>) -> Result<()> {
        for desc in descs {
            self.dispatch_compute(desc)?;
        }
        Ok(())
    }

    // ── Typed convenience methods (defaults via bytemuck) ─────────────

    /// Allocate a storage buffer for `n` f32 values.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn create_backend_buffer_f32(&self, n: usize) -> Result<Self::Buffer> {
        self.alloc_buffer("f32_buffer", (n * std::mem::size_of::<f32>()) as u64)
    }

    /// Allocate a storage buffer for `n` f64 values.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn create_backend_buffer_f64(&self, n: usize) -> Result<Self::Buffer> {
        self.alloc_buffer("f64_buffer", (n * std::mem::size_of::<f64>()) as u64)
    }

    /// Allocate a storage buffer for `n` u32 values.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn create_backend_buffer_u32(&self, n: usize) -> Result<Self::Buffer> {
        self.alloc_buffer("u32_buffer", (n * std::mem::size_of::<u32>()) as u64)
    }

    /// Allocate a storage buffer initialized with f32 data.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn alloc_buffer_f32_init(&self, label: &str, data: &[f32]) -> Result<Self::Buffer> {
        self.alloc_buffer_init(label, bytemuck::cast_slice(data))
    }

    /// Allocate a storage buffer initialized with f64 data.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn alloc_buffer_f64_init(&self, label: &str, data: &[f64]) -> Result<Self::Buffer> {
        self.alloc_buffer_init(label, bytemuck::cast_slice(data))
    }

    /// Write f32 data to a buffer at offset 0.
    fn upload_f32(&self, buffer: &Self::Buffer, data: &[f32]) {
        self.upload(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Write f64 data to a buffer at offset 0.
    fn upload_f64(&self, buffer: &Self::Buffer, data: &[f64]) {
        self.upload(buffer, 0, bytemuck::cast_slice(data));
    }

    /// Read `n` f32 values from a buffer.
    /// # Errors
    /// Returns [`Err`] if readback fails.
    fn download_f32(&self, buffer: &Self::Buffer, n: usize) -> Result<Vec<f32>> {
        let bytes = self.download(buffer, (n * std::mem::size_of::<f32>()) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Read `n` f64 values from a buffer.
    /// # Errors
    /// Returns [`Err`] if readback fails.
    fn download_f64(&self, buffer: &Self::Buffer, n: usize) -> Result<Vec<f64>> {
        let bytes = self.download(buffer, (n * std::mem::size_of::<f64>()) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Read `n` u32 values from a buffer.
    /// # Errors
    /// Returns [`Err`] if readback fails.
    fn download_u32(&self, buffer: &Self::Buffer, n: usize) -> Result<Vec<u32>> {
        let bytes = self.download(buffer, (n * std::mem::size_of::<u32>()) as u64)?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Allocate a uniform buffer from a `Pod` value.
    /// # Errors
    /// Returns [`Err`] if allocation fails.
    fn alloc_uniform_pod<T: bytemuck::NoUninit>(
        &self,
        label: &str,
        data: &T,
    ) -> Result<Self::Buffer> {
        self.alloc_uniform(label, bytemuck::bytes_of(data))
    }
}

/// Blanket implementation: `Arc<B>` delegates to the inner backend.
///
/// Most ops hold `Arc<WgpuDevice>` — this impl lets them pass `&self.device`
/// directly to `ComputeDispatch::new()` without explicit deref.
impl<B: GpuBackend> GpuBackend for Arc<B> {
    type Buffer = B::Buffer;

    fn name(&self) -> &str {
        (**self).name()
    }
    fn has_f64_shaders(&self) -> bool {
        (**self).has_f64_shaders()
    }
    fn is_lost(&self) -> bool {
        (**self).is_lost()
    }
    fn alloc_buffer(&self, label: &str, size: u64) -> Result<Self::Buffer> {
        (**self).alloc_buffer(label, size)
    }
    fn alloc_buffer_init(&self, label: &str, contents: &[u8]) -> Result<Self::Buffer> {
        (**self).alloc_buffer_init(label, contents)
    }
    fn alloc_uniform(&self, label: &str, contents: &[u8]) -> Result<Self::Buffer> {
        (**self).alloc_uniform(label, contents)
    }
    fn upload(&self, buffer: &Self::Buffer, offset: u64, data: &[u8]) {
        (**self).upload(buffer, offset, data);
    }
    fn download(&self, buffer: &Self::Buffer, size: u64) -> Result<bytes::Bytes> {
        (**self).download(buffer, size)
    }
    fn dispatch_compute(&self, desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        (**self).dispatch_compute(reborrow_descriptor(desc))
    }

    fn dispatch_compute_batch(&self, descs: Vec<DispatchDescriptor<'_, Self>>) -> Result<()> {
        let reborrowed = descs.into_iter().map(reborrow_descriptor).collect();
        (**self).dispatch_compute_batch(reborrowed)
    }
}

fn reborrow_descriptor<B: GpuBackend>(
    desc: DispatchDescriptor<'_, Arc<B>>,
) -> DispatchDescriptor<'_, B> {
    let bindings = desc
        .bindings
        .iter()
        .map(|b| BufferBinding {
            index: b.index,
            buffer: b.buffer,
            read_only: b.read_only,
            is_uniform: b.is_uniform,
        })
        .collect();
    DispatchDescriptor {
        label: desc.label,
        shader_source: desc.shader_source,
        entry_point: desc.entry_point,
        bindings,
        workgroups: desc.workgroups,
        f64_shader: desc.f64_shader,
        df64_shader: desc.df64_shader,
    }
}
