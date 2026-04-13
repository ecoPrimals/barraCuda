// SPDX-License-Identifier: AGPL-3.0-or-later
//! Buffer allocation and data transfer — generic `read_buffer`<T: Pod> + typed helpers

use super::WgpuDevice;
use crate::error::{BarracudaError, Result};

impl WgpuDevice {
    /// Read typed values from a GPU buffer.
    /// Creates a staging buffer, copies data, maps it, and extracts via bytemuck.
    /// All typed read variants delegate here.
    ///
    /// Uses the single-poll path: submit → `map_async` → one `poll_safe` — no
    /// double-poll.
    /// # Errors
    /// Returns [`Err`] if the device is lost before or during the readback copy,
    /// if [`poll_safe`] fails (e.g. device lost during poll), or if
    /// [`map_staging_buffer`] fails (device lost, buffer mapping channel closed,
    /// or wgpu `map_async` reports an error).
    pub(crate) fn read_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.is_lost() {
            return Err(BarracudaError::device_lost(
                "cannot read buffer — device lost",
            ));
        }
        let byte_size = (count * std::mem::size_of::<T>()) as u64;

        let _permit = self.acquire_dispatch();

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);

        self.submit_and_map::<T>(Some(encoder.finish()), &staging, count)
    }

    /// Submit command buffers then map and read a staging buffer in one pass.
    ///
    /// Collapses the old `submit_and_poll` → `map_staging_buffer` double-poll
    /// into a single submit → `map_async` → `poll_safe` cycle.  The single
    /// `poll_safe` processes both the compute/copy **and** the map callback.
    /// # Errors
    /// Returns [`Err`] if the device is lost, if [`poll_safe`] fails, if the
    /// buffer mapping channel closes, or if `map_async` reports an error.
    pub fn submit_and_map<T: bytemuck::Pod>(
        &self,
        commands: impl IntoIterator<Item = wgpu::CommandBuffer>,
        staging: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>> {
        if count == 0 {
            self.submit_commands(commands);
            return Ok(Vec::new());
        }
        if self.is_lost() {
            return Err(BarracudaError::device_lost(
                "cannot submit_and_map — device lost",
            ));
        }

        // Submit without polling — release the lock so other threads can
        // interleave their submits while we set up the map callback.
        self.submit_commands(commands);

        // Register the map callback before polling so that a single
        // poll(Wait) processes both the compute/copy AND the map request.
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.encoding_guard();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.encoding_complete();

        self.poll_safe()?;

        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        self.encoding_guard();
        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        self.encoding_complete();

        Ok(result)
    }

    /// Map a staging buffer and extract typed data.
    /// Use after you've already submitted a copy to the staging buffer
    /// and polled (e.g., via [`submit_and_poll_inner`]). This handles the
    /// `map_async` → `poll` → `get_mapped_range` → `unmap` dance that
    /// was previously duplicated across ~40 ops.
    ///
    /// Prefer [`submit_and_map`] for new code — it combines submission and
    /// mapping into a single call, avoiding double-poll overhead.
    /// # Errors
    /// Returns [`Err`] if the device is lost, if [`poll_safe`] fails (device lost
    /// during poll), if the GPU buffer mapping channel is closed before the
    /// callback runs, or if wgpu `map_async` reports an error.
    pub fn map_staging_buffer<T: bytemuck::Pod>(
        &self,
        staging: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<T>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.is_lost() {
            return Err(BarracudaError::device_lost(
                "cannot map buffer — device lost",
            ));
        }
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.encoding_guard();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.encoding_complete();

        self.poll_safe()?;

        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        self.encoding_guard();
        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        self.encoding_complete();

        Ok(result)
    }

    /// Read f64 values from a GPU buffer.
    /// This is the canonical readback path — all GPU ops should use this.
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`read_buffer`].
    pub fn read_f64_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f64>> {
        self.read_buffer::<f64>(buffer, count)
    }

    /// Read buffer to host memory (f32)
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`read_buffer`].
    pub fn read_buffer_f32(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<f32>> {
        self.read_buffer::<f32>(buffer, size)
    }

    /// Write data to buffer (f32)
    /// # Errors
    /// Never returns an error; always returns `Ok(())`.
    pub fn write_buffer_f32(&self, buffer: &wgpu::Buffer, data: &[f32]) -> Result<()> {
        self.encoding_guard();
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
        self.encoding_complete();
        Ok(())
    }

    /// Read f64 buffer to host memory
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`read_buffer`].
    pub fn read_buffer_f64(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<f64>> {
        self.read_buffer::<f64>(buffer, size)
    }

    /// Read u32 buffer to host memory
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`read_buffer`].
    pub fn read_buffer_u32(&self, buffer: &wgpu::Buffer, size: usize) -> Result<Vec<u32>> {
        self.read_buffer::<u32>(buffer, size)
    }

    /// Create storage buffer (convenience helper)
    #[must_use]
    pub fn create_storage_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        buf
    }

    /// Create uniform buffer (convenience helper)
    pub fn create_uniform_buffer<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        buf
    }

    /// Allocate buffer for f32 data.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the VRAM quota.
    pub fn create_buffer_f32(&self, size: usize) -> Result<wgpu::Buffer> {
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;
        self.quota_try_allocate(byte_size)?;
        self.encoding_guard();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("barraCuda Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encoding_complete();
        Ok(buf)
    }

    /// Create storage buffer initialized with u32 data.
    #[must_use]
    pub fn create_buffer_u32_init(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        buf
    }

    /// Allocate buffer for u32 data.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the VRAM quota.
    pub fn create_buffer_u32(&self, size: usize) -> Result<wgpu::Buffer> {
        let byte_size = (size * std::mem::size_of::<u32>()) as u64;
        self.quota_try_allocate(byte_size)?;
        self.encoding_guard();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("barraCuda U32 Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encoding_complete();
        Ok(buf)
    }

    /// Allocate zero-initialized buffer for u32 data.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the VRAM quota.
    pub fn create_buffer_u32_zeros(&self, size: usize) -> Result<wgpu::Buffer> {
        let byte_size = (size * std::mem::size_of::<u32>()) as u64;
        self.quota_try_allocate(byte_size)?;
        self.encoding_guard();
        let zeros = vec![0u32; size];
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("barraCuda U32 Zeros Buffer"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        self.encoding_complete();
        Ok(buf)
    }

    /// Create storage buffer initialized with f32 data.
    #[must_use]
    pub fn create_buffer_f32_init(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        buf
    }

    /// Write data to buffer (f64)
    /// # Errors
    /// Never returns an error; always returns `Ok(())`.
    pub fn write_buffer_f64(&self, buffer: &wgpu::Buffer, data: &[f64]) -> Result<()> {
        self.encoding_guard();
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
        self.encoding_complete();
        Ok(())
    }

    /// Create storage buffer initialized with f64 data.
    #[must_use]
    pub fn create_buffer_f64_init(&self, label: &str, data: &[f64]) -> wgpu::Buffer {
        self.encoding_guard();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        self.encoding_complete();
        buf
    }

    /// Allocate buffer for f64 data.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the VRAM quota.
    pub fn create_buffer_f64(&self, size: usize) -> Result<wgpu::Buffer> {
        let byte_size = (size * std::mem::size_of::<f64>()) as u64;
        self.quota_try_allocate(byte_size)?;
        self.encoding_guard();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("barraCuda F64 Buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encoding_complete();
        Ok(buf)
    }

    // ── f32 buffer ergonomic helpers (hotSpring ESN GPU dispatch absorption) ───

    /// Create an f32 read-write storage buffer (GPU shader ↔ CPU).
    /// Equivalent to `create_buffer_f32` with a descriptive label.
    /// For ESN and other f32 GPU pipelines.
    /// # Errors
    /// Returns [`Err`] if the allocation would exceed the VRAM quota.
    pub fn create_f32_rw_buffer(&self, label: &str, count: usize) -> Result<wgpu::Buffer> {
        let byte_size = (count * std::mem::size_of::<f32>()) as u64;
        self.quota_try_allocate(byte_size)?;
        self.encoding_guard();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.encoding_complete();
        Ok(buf)
    }

    /// Create an f32 output buffer (GPU write → CPU read).
    /// Like `create_f32_rw_buffer` but labelled for clarity in GPU dispatch
    /// pipelines where the buffer is only written by the shader and read back
    /// to the host.
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`create_f32_rw_buffer`].
    pub fn create_f32_output_buffer(&self, label: &str, count: usize) -> Result<wgpu::Buffer> {
        self.create_f32_rw_buffer(label, count)
    }

    /// Upload f32 data to a new GPU buffer in one call.
    /// Creates and initializes a labeled storage buffer. For ESN weight
    /// uploads and similar patterns.
    #[must_use]
    pub fn upload_f32(&self, label: &str, data: &[f32]) -> wgpu::Buffer {
        self.create_buffer_f32_init(label, data)
    }

    /// Read back f32 data from a GPU buffer (GPU → CPU).
    /// Ergonomic alias for `read_buffer_f32`.
    /// # Errors
    /// Returns [`Err`] with the same conditions as [`read_buffer`].
    pub fn read_back_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        self.read_buffer_f32(buffer, count)
    }

    /// Minimal 8-byte storage buffer for unused **read-only** bind-group slots.
    ///
    /// WGSL and the WebGPU spec require every binding in a bind-group layout
    /// to be populated when creating a bind group — there is no optional-binding
    /// mechanism. When a shader entry point doesn't use all slots (e.g., a
    /// velocity half-step that ignores position output, or shared layouts with
    /// multiple entry points), callers must provide a valid buffer for each slot.
    /// This method returns a singleton placeholder to avoid ad-hoc dummy buffers
    /// at every call site.
    ///
    /// WebGPU forbids aliasing the same buffer across `storage_read` and
    /// `storage_rw` bindings, so use [`placeholder_buffer_rw`] for unused
    /// read-write slots.
    pub fn placeholder_buffer(&self) -> &wgpu::Buffer {
        static PLACEHOLDER_RO: std::sync::OnceLock<wgpu::Buffer> = std::sync::OnceLock::new();
        PLACEHOLDER_RO.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("placeholder:unused_read_slot"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
    }

    /// Minimal 8-byte storage buffer for unused **read-write** bind-group slots.
    ///
    /// Separate from [`placeholder_buffer`] to avoid WebGPU aliasing violations
    /// when both read and read-write slots are unused in the same dispatch.
    /// See [`placeholder_buffer`] for the WGSL/WebGPU rationale.
    pub fn placeholder_buffer_rw(&self) -> &wgpu::Buffer {
        static PLACEHOLDER_RW: std::sync::OnceLock<wgpu::Buffer> = std::sync::OnceLock::new();
        PLACEHOLDER_RW.get_or_init(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("placeholder:unused_rw_slot"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
    }
}
