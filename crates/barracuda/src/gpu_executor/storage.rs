// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU tensor storage for the `ComputeExecutor` scheduler interface.
//!
//! Holds a real `wgpu::Buffer` so that `read_to_cpu` / `write_from_cpu`
//! and `GpuExecutor::execute()` all operate on the same underlying GPU memory.
//!
//! The buffer is stored as `Arc<wgpu::Buffer>` so it can be shared with a
//! `Tensor` via `Tensor::from_arc_buffer` — eliminating the GPU→CPU→GPU
//! round-trip when wrapping an executed output back into TensorStorage.

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::unified_hardware::{HardwareType, TensorStorage};
use crate::unified_math::{DType, TensorDescriptor};
use async_trait::async_trait;
use std::sync::Arc;

/// GPU tensor storage for the `ComputeExecutor` scheduler interface.
///
/// Holds a real `wgpu::Buffer` so that `read_to_cpu` / `write_from_cpu`
/// and `GpuExecutor::execute()` all operate on the same underlying GPU memory.
///
/// The buffer is stored as `Arc<wgpu::Buffer>` so it can be shared with a
/// `Tensor` via `Tensor::from_arc_buffer` — eliminating the GPU→CPU→GPU
/// round-trip when wrapping an executed output back into TensorStorage.
pub(crate) struct GpuTensorStorage {
    pub(crate) descriptor: TensorDescriptor,
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) buffer: Arc<wgpu::Buffer>,
}

impl GpuTensorStorage {
    pub(crate) fn new(descriptor: TensorDescriptor, device: Arc<WgpuDevice>) -> Self {
        let byte_size = (descriptor.numel * descriptor.dtype.size_bytes()) as u64;
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensorStorage"),
            size: byte_size.max(4), // wgpu requires size ≥ 4
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            descriptor,
            device,
            buffer: Arc::new(buffer),
        }
    }

    /// Zero-copy construction from a `Tensor` output.
    ///
    /// Shares the tensor's underlying `Arc<wgpu::Buffer>` — no data movement.
    /// Falls back to allocation + upload when the tensor uses a pooled buffer
    /// (pooled buffers may be reclaimed; we must own the buffer to guarantee
    /// `read_to_cpu` safety).
    pub(crate) fn from_tensor(tensor: &crate::tensor::Tensor, dtype: DType) -> Self {
        let shape = tensor.shape().to_vec();
        let numel: usize = shape.iter().product();
        let desc = TensorDescriptor::new(shape, dtype);

        if let Some(arc) = tensor.try_arc_buffer() {
            // Fast path: share the buffer, zero copies.
            Self {
                descriptor: desc,
                device: tensor.device().clone(),
                buffer: arc,
            }
        } else {
            // Pooled buffer: allocate our own storage and copy.
            let new = Self::new(desc, tensor.device().clone());
            // Synchronous upload — the Tensor's content is already on GPU;
            // copy_buffer_to_buffer moves it without touching the CPU.
            let byte_size = (numel * dtype.size_bytes()) as u64;
            let mut enc =
                new.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("GpuTensorStorage copy"),
                    });
            enc.copy_buffer_to_buffer(tensor.buffer(), 0, &new.buffer, 0, byte_size);
            new.device.submit_and_poll(Some(enc.finish()));
            new
        }
    }
}

// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
impl TensorStorage for GpuTensorStorage {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::GPU
    }

    /// Read GPU data back to CPU as raw bytes.
    /// Zero-copy access to the GPU buffer — enables callers to skip the
    /// GPU→CPU→GPU round-trip when the buffer is already on the right device.
    fn as_wgpu_buffer(&self) -> Option<Arc<wgpu::Buffer>> {
        Some(self.buffer.clone())
    }

    async fn read_to_cpu(&self) -> Result<Vec<u8>> {
        let numel = self.descriptor.numel;
        let elem_size = self.descriptor.dtype.size_bytes();
        let byte_size = (numel * elem_size) as u64;

        // Staging buffer for map-read
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensorStorage read staging"),
            size: byte_size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GpuTensorStorage read"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_size);
        self.device.submit_and_poll(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| crate::error::BarracudaError::Gpu("map_async channel closed".to_string()))?
            .map_err(|e| crate::error::BarracudaError::Gpu(format!("Buffer map failed: {e:?}")))?;

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(data)
    }

    /// Upload raw bytes from CPU to the GPU buffer.
    async fn write_from_cpu(&mut self, data: &[u8]) -> Result<()> {
        self.device.queue.write_buffer(&self.buffer, 0, data);
        Ok(())
    }
}
