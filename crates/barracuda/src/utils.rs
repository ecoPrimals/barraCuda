// SPDX-License-Identifier: AGPL-3.0-or-later
//! Utility functions for barracuda operations

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// Convert a byte slice to a fixed-size array. Used for `chunks_exact(N)` or
/// fixed-size buffers where the slice length is guaranteed to be N.
#[inline]
pub(crate) fn chunk_to_array<const N: usize>(chunk: &[u8]) -> Result<[u8; N]> {
    chunk.try_into().map_err(|_| BarracudaError::InvalidInput {
        message: format!("byte slice length {} != expected {}", chunk.len(), N),
    })
}

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
