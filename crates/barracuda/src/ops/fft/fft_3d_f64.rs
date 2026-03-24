// SPDX-License-Identifier: AGPL-3.0-or-later
//! 3D Fast Fourier Transform — Batched GPU Dispatch (f64 Precision)
//!
//! Performs 3D FFT via dimension-wise decomposition with a single batched
//! shader per axis. Uses the shared engine from `fft_1d` (`dispatch_axis_f64`,
//! `upload_twiddles_f64`, `AxisConfig`, `BATCHED_SHADER_F64`).
//!
//! Reads back via `dev.read_f64_buffer(&buf_a, expected_len)`.

use super::fft_1d::{AxisConfig, dispatch_axis_f64, upload_twiddles_f64};
use crate::error::{BarracudaError, Result};
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

/// 3D Complex FFT operation (f64 precision) — batched GPU dispatch
pub struct Fft3DF64 {
    device: Arc<crate::device::WgpuDevice>,
    nx: usize,
    ny: usize,
    nz: usize,
    twiddles: HashMap<u32, (wgpu::Buffer, wgpu::Buffer)>,
}

impl Fft3DF64 {
    /// Create a new 3D FFT operation.
    /// # Errors
    /// Returns [`Err`] if any of `nx`, `ny`, or `nz` are not powers of 2.
    pub fn new(
        device: Arc<crate::device::WgpuDevice>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self> {
        if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
            return Err(BarracudaError::InvalidInput {
                message: format!("FFT 3D dimensions must be powers of 2, got ({nx}, {ny}, {nz})"),
            });
        }

        let mut twiddles = HashMap::new();
        for &n in &[nx, ny, nz] {
            let n32 = n as u32;
            twiddles
                .entry(n32)
                .or_insert_with(|| upload_twiddles_f64(device.as_ref(), n32));
        }

        Ok(Self {
            device,
            nx,
            ny,
            nz,
            twiddles,
        })
    }

    /// Compute forward 3D FFT (time → frequency domain).
    /// # Errors
    /// Returns [`Err`] if data length mismatches or GPU dispatch fails.
    pub async fn forward(&self, data: &[f64]) -> Result<Vec<f64>> {
        self.execute_internal(data, false).await
    }

    /// Compute inverse 3D FFT (frequency → time domain).
    /// # Errors
    /// Returns [`Err`] if data length mismatches or GPU dispatch fails.
    pub async fn inverse(&self, data: &[f64]) -> Result<Vec<f64>> {
        self.execute_internal(data, true).await
    }

    async fn execute_internal(&self, data: &[f64], inverse: bool) -> Result<Vec<f64>> {
        let size = self.nx * self.ny * self.nz;
        let expected_len = size * 2;

        if data.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "FFT 3D data length {} doesn't match expected {} ({}x{}x{}x2)",
                    data.len(),
                    expected_len,
                    self.nx,
                    self.ny,
                    self.nz
                ),
            });
        }

        let buffer_bytes = (expected_len * size_of::<f64>()) as u64;
        let dev = &self.device;

        let buf_a = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FFT3D buf_a"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let buf_b = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT3D buf_b"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let axes = [
            AxisConfig {
                degree: self.nz as u32,
                element_stride: 1,
                dim1_stride: (self.ny * self.nz) as u32,
                dim2_stride: self.nz as u32,
                dim1_count: self.nx as u32,
                dim2_count: self.ny as u32,
            },
            AxisConfig {
                degree: self.ny as u32,
                element_stride: self.nz as u32,
                dim1_stride: (self.ny * self.nz) as u32,
                dim2_stride: 1,
                dim1_count: self.nx as u32,
                dim2_count: self.nz as u32,
            },
            AxisConfig {
                degree: self.nx as u32,
                element_stride: (self.ny * self.nz) as u32,
                dim1_stride: self.nz as u32,
                dim2_stride: 1,
                dim1_count: self.ny as u32,
                dim2_count: self.nz as u32,
            },
        ];

        for axis in &axes {
            let (tw_re, tw_im) = &self.twiddles[&axis.degree];
            dispatch_axis_f64(
                dev,
                &buf_a,
                &buf_b,
                buffer_bytes,
                tw_re,
                tw_im,
                axis,
                inverse,
            )?;
        }

        dev.read_f64_buffer(&buf_a, expected_len)
    }

    /// Return (nx, ny, nz) dimensions.
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fft_3d_f64_roundtrip() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let n = 4;
        let size = n * n * n;

        let mut data = vec![0.0f64; size * 2];
        data[0] = 1.0;

        let fft = Fft3DF64::new(device.clone(), n, n, n).unwrap();

        let freq = fft.forward(&data).await.unwrap();
        assert_eq!(freq.len(), size * 2);

        for i in 0..size {
            let re = freq[i * 2];
            let im = freq[i * 2 + 1];
            let mag = re.hypot(im);
            assert!(
                (mag - 1.0).abs() < 1e-10,
                "Expected magnitude 1.0, got {mag}"
            );
        }

        let back = fft.inverse(&freq).await.unwrap();

        let norm = (size as f64).recip();
        let back_norm: Vec<f64> = back.iter().map(|x| x * norm).collect();

        assert!((back_norm[0] - 1.0).abs() < 1e-10);
        for i in 1..size {
            assert!((back_norm[i * 2]).abs() < 1e-10);
            assert!((back_norm[i * 2 + 1]).abs() < 1e-10);
        }
    }

    #[tokio::test]
    async fn test_fft_3d_f64_sinusoidal_roundtrip() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let n = 8;
        let size = n * n * n;
        let pi = std::f64::consts::PI;

        let mut data = vec![0.0f64; size * 2];
        for ix in 0..n {
            for iy in 0..n {
                for iz in 0..n {
                    let idx = (ix * n * n + iy * n + iz) * 2;
                    data[idx] = 0.5f64.mul_add(
                        (2.0 * pi * iz as f64 / n as f64).cos(),
                        (2.0 * pi * ix as f64 / n as f64).sin(),
                    );
                }
            }
        }
        let original = data.clone();

        let fft = Fft3DF64::new(device.clone(), n, n, n).unwrap();
        let freq = fft.forward(&data).await.unwrap();
        let back = fft.inverse(&freq).await.unwrap();

        let norm = (size as f64).recip();
        for i in 0..size {
            let re = back[i * 2] * norm;
            let im = back[i * 2 + 1] * norm;
            assert!(
                (re - original[i * 2]).abs() < 1e-8,
                "Voxel {} real: expected {:.15e}, got {re:.15e}",
                i,
                original[i * 2]
            );
            assert!(
                im.abs() < 1e-8,
                "Voxel {i} imag: expected ~0, got {im:.15e}"
            );
        }
    }

    #[tokio::test]
    async fn test_fft_3d_f64_non_cubic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let (nx, ny, nz) = (4, 8, 16);
        let size = nx * ny * nz;
        let mut data = vec![0.0f64; size * 2];
        data[0] = 1.0;

        let fft = Fft3DF64::new(device.clone(), nx, ny, nz).unwrap();
        let freq = fft.forward(&data).await.unwrap();
        let back = fft.inverse(&freq).await.unwrap();

        let norm = (size as f64).recip();
        assert!(back[0].mul_add(norm, -1.0).abs() < 1e-10);
        for i in 1..size {
            assert!((back[i * 2] * norm).abs() < 1e-10);
        }
    }
}
