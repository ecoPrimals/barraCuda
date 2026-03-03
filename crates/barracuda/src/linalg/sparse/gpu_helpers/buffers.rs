//! Buffer creation and I/O for sparse GPU solvers.
//!
//! Single responsibility: buffer allocation, readback, and copy operations.
//! Reused by CG, BiCGSTAB, and other sparse solvers.

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Buffer creation helpers for sparse GPU solvers
pub struct SparseBuffers;

impl SparseBuffers {
    /// Create an f64 storage buffer from data
    pub fn f64_from_slice(device: &Arc<WgpuDevice>, label: &str, data: &[f64]) -> wgpu::Buffer {
        Self::f64_from_slice_raw(&device.device, label, data)
    }

    /// Create a zero-initialized f64 storage buffer
    pub fn f64_zeros(device: &Arc<WgpuDevice>, label: &str, count: usize) -> wgpu::Buffer {
        Self::f64_zeros_raw(&device.device, label, count)
    }

    /// Create an f64 storage buffer from slice (raw device)
    pub fn f64_from_slice_raw(device: &wgpu::Device, label: &str, data: &[f64]) -> wgpu::Buffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a zero-initialized f64 storage buffer (raw device)
    pub fn f64_zeros_raw(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
        let zeros = vec![0u8; count * 8];
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &zeros,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Create a zero-initialized i32 storage buffer (raw device)
    pub fn i32_zeros_raw(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Poll a raw wgpu::Device with device-lost protection.
    fn poll_raw_safe(device: &wgpu::Device) -> Result<()> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.poll(wgpu::Maintain::Wait);
        }));
        result.map_err(|payload| {
            let msg = payload
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("unknown");
            BarracudaError::device(format!("GPU device lost during poll: {msg}"))
        })
    }

    /// Read f64 data from GPU buffer (raw device/queue)
    pub fn read_f64_raw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("f64 staging"),
            size: (count * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("f64 readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 8) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        Self::poll_raw_safe(device)?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<f64> = data
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(8) yields 8-byte chunks"),
                )
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Read i32 data from GPU buffer (raw device/queue)
    pub fn read_i32_raw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<i32>> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("i32 staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("i32 readback"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        Self::poll_raw_safe(device)?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<i32> = data
            .chunks_exact(4)
            .map(|chunk| {
                i32::from_le_bytes(
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) yields 4-byte chunks"),
                )
            })
            .collect();
        drop(data);
        staging.unmap();

        Ok(result)
    }

    /// Create a u32 storage buffer from usize data (for CSR indices)
    pub fn u32_from_usize(device: &Arc<WgpuDevice>, label: &str, data: &[usize]) -> wgpu::Buffer {
        let u32_data: Vec<u32> = data.iter().map(|&x| x as u32).collect();
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&u32_data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Create a uniform buffer from u32 params
    pub fn uniform_u32(device: &Arc<WgpuDevice>, label: &str, params: &[u32]) -> wgpu::Buffer {
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Read f64 data from GPU buffer
    pub fn read_f64(
        device: &Arc<WgpuDevice>,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>> {
        Self::read_f64_raw(&device.device, &device.queue, buffer, count)
    }

    /// Write f64 data to GPU buffer
    pub fn write_f64(device: &Arc<WgpuDevice>, buffer: &wgpu::Buffer, data: &[f64]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.queue.write_buffer(buffer, 0, &bytes);
    }

    /// Copy buffer to buffer (f64)
    pub fn copy_f64(
        device: &Arc<WgpuDevice>,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        count: usize,
    ) {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Buffer copy"),
            });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, (count * 8) as u64);
        device.submit_and_poll(Some(encoder.finish()));
    }
}
