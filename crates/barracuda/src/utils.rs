// SPDX-License-Identifier: AGPL-3.0-or-later
//! Utility functions for barracuda operations

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// Read f32 buffer data from GPU back to CPU.
/// Delegates to `WgpuDevice::read_buffer` which holds the submit guard.
///
/// # Errors
///
/// Returns [`Err`] if buffer readback fails (e.g., device lost, `map_async` error).
pub fn read_buffer(
    device: &Arc<WgpuDevice>,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<f32>> {
    device.read_buffer_f32(buffer, size)
}

/// Read u32 buffer data from GPU back to CPU.
/// Delegates to `WgpuDevice::read_buffer` which holds the submit guard.
///
/// # Errors
///
/// Returns [`Err`] if buffer readback fails (e.g., device lost, `map_async` error).
pub fn read_buffer_u32(
    device: &Arc<WgpuDevice>,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<u32>> {
    device.read_buffer_u32(buffer, size)
}

/// Read f64 buffer data from GPU back to CPU.
/// Delegates to `WgpuDevice::read_buffer` which holds the submit guard.
///
/// # Errors
///
/// Returns [`Err`] if buffer readback fails (e.g., device lost, `map_async` error).
pub fn read_buffer_f64(
    device: &Arc<WgpuDevice>,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<Vec<f64>> {
    device.read_buffer_f64(buffer, size)
}
