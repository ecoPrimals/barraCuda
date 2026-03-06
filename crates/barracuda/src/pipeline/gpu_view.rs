// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GpuView<T>` — Typed handle to GPU-resident data.
//!
//! Eliminates per-call host↔device round-trips by keeping data on the GPU
//! between operations.  Springs chain `GpuView` references through fused
//! reduction and transform stages without any CPU involvement.
//!
//! # The Kokkos Pattern (groundSpring V74 benchmark, Mar 4 2026)
//!
//! Kokkos keeps `View<double*>` in device memory across operations.  A
//! `parallel_reduce` over 1M doubles is a single kernel launch with near-zero
//! overhead.  barraCuda's previous pattern required a full host→device→host
//! round-trip per function call — 100×–2,600× slower on statistical reductions.
//!
//! `GpuView<T>` mirrors the Kokkos pattern in safe Rust:
//!
//! ```text
//! let view = GpuViewF64::upload(&device, &data)?;   // host → device (once)
//! let mean_var = view.mean_variance(0)?;              // GPU-only dispatch
//! let corr = GpuViewF64::correlation(&view_x, &view_y)?; // GPU-only dispatch
//! let result = view.download()?;                      // device → host (when needed)
//! ```
//!
//! # Performance Target
//!
//! | Metric            | Before (per-call) | After (GpuView) | Improvement |
//! |-------------------|-------------------|-----------------|-------------|
//! | mean (1M f64)     | 8,454 µs          | < 100 µs        | > 80×       |
//! | variance (1M f64) | 8,515 µs          | < 100 µs        | > 80×       |
//! | Pearson r (1M)    | 125 ms            | < 200 µs        | > 600×      |

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::marker::PhantomData;
use std::sync::Arc;

/// Typed handle to a GPU-resident buffer.
///
/// Data stays on the device until explicitly downloaded.  All pipeline
/// operations accept `&GpuView<T>` references — no implicit transfers.
pub struct GpuView<T: GpuViewElement> {
    device: Arc<WgpuDevice>,
    buffer: wgpu::Buffer,
    len: usize,
    _phantom: PhantomData<T>,
}

/// Trait for types that can live in a `GpuView`.
pub trait GpuViewElement: Copy + bytemuck::Pod {
    /// Size of one element in bytes.
    const ELEMENT_SIZE: usize;
}

impl GpuViewElement for f64 {
    const ELEMENT_SIZE: usize = 8;
}

impl GpuViewElement for f32 {
    const ELEMENT_SIZE: usize = 4;
}

impl GpuViewElement for u32 {
    const ELEMENT_SIZE: usize = 4;
}

impl GpuViewElement for i32 {
    const ELEMENT_SIZE: usize = 4;
}

impl<T: GpuViewElement> GpuView<T> {
    /// Upload host data to a new GPU-resident view.
    /// # Errors
    /// Returns [`Err`] if `data` is empty or buffer allocation fails.
    pub fn upload(device: Arc<WgpuDevice>, data: &[T]) -> Result<Self> {
        if data.is_empty() {
            return Err(BarracudaError::InvalidInput {
                message: "GpuView: cannot create from empty data".into(),
            });
        }
        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GpuView"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Ok(Self {
            device,
            buffer,
            len: data.len(),
            _phantom: PhantomData,
        })
    }

    /// Create an uninitialized GPU-resident view of `len` elements.
    /// Useful as an output target for GPU kernels.
    /// # Errors
    /// Returns [`Err`] if `len` is zero or buffer allocation fails.
    pub fn uninit(device: Arc<WgpuDevice>, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "GpuView: cannot create zero-length view".into(),
            });
        }
        let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuView:uninit"),
            size: (len * T::ELEMENT_SIZE) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            buffer,
            len,
            _phantom: PhantomData,
        })
    }

    /// Download data from GPU to host.
    /// # Errors
    /// Returns [`Err`] if the device is lost during poll, the staging buffer
    /// `map_async` callback fails, or the channel is closed.
    pub fn download(&self) -> Result<Vec<T>> {
        let byte_size = (self.len * T::ELEMENT_SIZE) as u64;
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuView:download:staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuView:download"),
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
            .map_err(|_| BarracudaError::execution_failed("GpuView: staging channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }

    /// Overwrite GPU-resident data from host.
    /// # Errors
    /// Returns [`Err`] if `data.len()` does not match the view length.
    pub fn upload_into(&self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "GpuView: upload_into length {} != view length {}",
                    data.len(),
                    self.len
                ),
            });
        }
        self.device
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// Number of elements in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view is empty (always false after construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        (self.len * T::ELEMENT_SIZE) as u64
    }

    /// Reference to the underlying wgpu buffer (for binding in compute passes).
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Reference to the device.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

/// Convenience alias for f64 GPU views.
pub type GpuViewF64 = GpuView<f64>;
/// Convenience alias for f32 GPU views.
pub type GpuViewF32 = GpuView<f32>;
/// Convenience alias for u32 GPU views.
pub type GpuViewU32 = GpuView<u32>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_gpu_device;

    #[tokio::test]
    async fn test_gpu_view_upload_download_roundtrip() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let view = GpuViewF64::upload(device, &data).unwrap();
        assert_eq!(view.len(), 5);

        let result = view.download().unwrap();
        for (i, (&got, &expected)) in result.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-15,
                "element[{i}] = {got}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_gpu_view_upload_into() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let initial = vec![0.0_f64; 4];
        let view = GpuViewF64::upload(Arc::clone(&device), &initial).unwrap();

        let new_data = vec![7.0, 8.0, 9.0, 10.0];
        view.upload_into(&new_data).unwrap();

        let result = view.download().unwrap();
        assert_eq!(result, new_data);
    }

    #[tokio::test]
    async fn test_gpu_view_uninit() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let view = GpuViewF64::uninit(device, 100).unwrap();
        assert_eq!(view.len(), 100);
        assert_eq!(view.byte_size(), 800);
    }

    #[tokio::test]
    async fn test_gpu_view_f32() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let data = vec![1.0_f32, 2.0, 3.0];
        let view = GpuViewF32::upload(device, &data).unwrap();
        let result = view.download().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_gpu_view_element_sizes() {
        assert_eq!(f64::ELEMENT_SIZE, 8);
        assert_eq!(f32::ELEMENT_SIZE, 4);
        assert_eq!(u32::ELEMENT_SIZE, 4);
    }
}
