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

/// Checked conversion from `usize` to `u32` for GPU uniform/dispatch values.
///
/// Returns `BarracudaError::InvalidInput` if the value exceeds `u32::MAX`,
/// preventing silent truncation for large tensors.
#[inline]
pub(crate) fn checked_u32(value: usize, context: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| BarracudaError::InvalidInput {
        message: format!(
            "{context}: {value} exceeds u32::MAX ({max})",
            max = u32::MAX
        ),
    })
}

/// Convert a shape slice to `Vec<u32>` for GPU uniform buffers.
///
/// Returns `BarracudaError::InvalidInput` if any dimension exceeds `u32::MAX`.
#[inline]
#[expect(
    dead_code,
    reason = "public API for incremental migration of shape casts"
)]
pub(crate) fn shape_to_u32(dims: &[usize]) -> Result<Vec<u32>> {
    dims.iter()
        .map(|&d| checked_u32(d, "shape dimension"))
        .collect()
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

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "test code")]

    use super::*;
    use crate::error::BarracudaError;

    #[test]
    fn chunk_to_array_exact_length_succeeds() {
        let chunk: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let arr = chunk_to_array::<8>(&chunk).unwrap();
        assert_eq!(arr, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn chunk_to_array_too_short_returns_err() {
        let chunk: [u8; 4] = [1, 2, 3, 4];
        let err = chunk_to_array::<8>(&chunk).unwrap_err();
        match &err {
            BarracudaError::InvalidInput { message } => {
                assert!(message.contains('4'));
                assert!(message.contains('8'));
            }
            _ => panic!("expected InvalidInput, got {err:?}"),
        }
    }

    #[test]
    fn chunk_to_array_too_long_returns_err() {
        let chunk: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let err = chunk_to_array::<8>(&chunk).unwrap_err();
        match &err {
            BarracudaError::InvalidInput { message } => {
                assert!(message.contains("12"));
                assert!(message.contains('8'));
            }
            _ => panic!("expected InvalidInput, got {err:?}"),
        }
    }

    #[test]
    fn chunk_to_array_empty_returns_err() {
        let chunk: [u8; 0] = [];
        let err = chunk_to_array::<1>(chunk.as_slice()).unwrap_err();
        match &err {
            BarracudaError::InvalidInput { message } => {
                assert!(message.contains('0'));
                assert!(message.contains('1'));
            }
            _ => panic!("expected InvalidInput, got {err:?}"),
        }
    }

    #[test]
    fn chunk_to_array_size_one() {
        let chunk = [42u8];
        let arr = chunk_to_array::<1>(&chunk).unwrap();
        assert_eq!(arr, [42]);
    }
}
