// SPDX-License-Identifier: AGPL-3.0-or-later
//! Buffer creation and I/O for sparse GPU solvers.
//!
//! Single responsibility: buffer allocation, readback, and copy operations.
//! Reused by CG, BiCGSTAB, and other sparse solvers.

use crate::device::WgpuDevice;
use crate::error::Result;
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

    /// Read f64 data from GPU buffer (synchronized via WgpuDevice)
    pub fn read_f64_raw(
        device: &WgpuDevice,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>> {
        device.read_buffer::<f64>(buffer, count)
    }

    /// Read i32 data from GPU buffer (synchronized via WgpuDevice)
    pub fn read_i32_raw(
        device: &WgpuDevice,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<i32>> {
        device.read_buffer::<i32>(buffer, count)
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
        Self::read_f64_raw(device.as_ref(), buffer, count)
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
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Buffer copy"),
        });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, (count * 8) as u64);
        device.submit_and_poll(Some(encoder.finish()));
    }
}
