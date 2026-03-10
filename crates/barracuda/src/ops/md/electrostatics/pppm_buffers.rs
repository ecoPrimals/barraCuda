// SPDX-License-Identifier: AGPL-3.0-only
//! PPPM GPU buffer helpers
//!
//! Extracted from `pppm_gpu.rs` for modularity (Feb 14, 2026).
//! Delegates to `crate::linalg::sparse::SparseBuffers` for shared implementation (Feb 15, 2026).

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::linalg::sparse::SparseBuffers;
use wgpu::util::DeviceExt;

/// PPPM buffer utilities for GPU memory management
///
/// Thin wrapper over `SparseBuffers` for electrostatics code paths that use raw
/// `wgpu::Device` and `wgpu::Queue` (e.g. `PppmGpu`).
pub struct PppmBuffers;

impl PppmBuffers {
    /// Create f64 buffer initialized with data
    #[must_use]
    pub fn f64_from_slice(device: &wgpu::Device, label: &str, data: &[f64]) -> wgpu::Buffer {
        SparseBuffers::f64_from_slice_raw(device, label, data)
    }

    /// Create zero-initialized f64 buffer
    #[must_use]
    pub fn f64_zeros(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
        SparseBuffers::f64_zeros_raw(device, label, count)
    }

    /// Create zero-initialized i32 buffer
    #[must_use]
    pub fn i32_zeros(device: &wgpu::Device, label: &str, count: usize) -> wgpu::Buffer {
        SparseBuffers::i32_zeros_raw(device, label, count)
    }

    /// Create i32 buffer from slice
    #[must_use]
    pub fn i32_from_slice(device: &wgpu::Device, label: &str, data: &[i32]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// Read f64 buffer back to CPU (sync; async wrapper for API compatibility)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub async fn read_f64(
        device: &WgpuDevice,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<f64>> {
        SparseBuffers::read_f64_raw(device, buffer, count)
    }

    /// Read i32 buffer back to CPU (sync; async wrapper for API compatibility)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub async fn read_i32(
        device: &WgpuDevice,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Result<Vec<i32>> {
        SparseBuffers::read_i32_raw(device, buffer, count)
    }
}

/// CPU FFT utilities for PPPM (fallback when GPU FFT not available)
pub struct PppmCpuFft;

impl PppmCpuFft {
    /// Forward 3D FFT (real input → complex output)
    #[must_use]
    pub fn forward_3d(mesh: &[f64], kx: usize, ky: usize, kz: usize) -> Vec<f64> {
        let size = kx * ky * kz;

        // Convert real mesh to complex
        let mut complex = vec![0.0f64; size * 2];
        for i in 0..size {
            complex[i * 2] = mesh[i];
        }

        // 3D FFT via 1D transforms
        Self::fft_3d(&mut complex, kx, ky, kz, false);

        complex
    }

    /// Inverse 3D FFT (complex input → real output, normalized)
    #[must_use]
    pub fn inverse_3d(phi_k: &[f64], kx: usize, ky: usize, kz: usize) -> Vec<f64> {
        let size = kx * ky * kz;

        let mut complex = phi_k.to_vec();
        Self::fft_3d(&mut complex, kx, ky, kz, true);

        // Extract real part (fft_3d already applied 1/N normalization)
        (0..size).map(|i| complex[i * 2]).collect()
    }

    /// 3D FFT via 1D transforms along each axis
    ///
    /// Matches CPU Pppm's `fft_3d_cpu` exactly so GPU path produces same `e_kspace/forces`.
    fn fft_3d(data: &mut [f64], nx: usize, ny: usize, nz: usize, inverse: bool) {
        // FFT along Z (innermost)
        for ix in 0..nx {
            for iy in 0..ny {
                let mut pencil = vec![0.0; nz * 2];
                for iz in 0..nz {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[iz * 2] = data[idx];
                    pencil[iz * 2 + 1] = data[idx + 1];
                }

                Self::fft_1d(&mut pencil, nz, inverse);

                for iz in 0..nz {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[iz * 2];
                    data[idx + 1] = pencil[iz * 2 + 1];
                }
            }
        }

        // FFT along Y
        for ix in 0..nx {
            for iz in 0..nz {
                let mut pencil = vec![0.0; ny * 2];
                for iy in 0..ny {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[iy * 2] = data[idx];
                    pencil[iy * 2 + 1] = data[idx + 1];
                }

                Self::fft_1d(&mut pencil, ny, inverse);

                for iy in 0..ny {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[iy * 2];
                    data[idx + 1] = pencil[iy * 2 + 1];
                }
            }
        }

        // FFT along X (outermost)
        for iy in 0..ny {
            for iz in 0..nz {
                let mut pencil = vec![0.0; nx * 2];
                for ix in 0..nx {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[ix * 2] = data[idx];
                    pencil[ix * 2 + 1] = data[idx + 1];
                }

                Self::fft_1d(&mut pencil, nx, inverse);

                for ix in 0..nx {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[ix * 2];
                    data[idx + 1] = pencil[ix * 2 + 1];
                }
            }
        }

        // Normalize for inverse FFT
        if inverse {
            let scale = 1.0 / (nx * ny * nz) as f64;
            for v in data.iter_mut() {
                *v *= scale;
            }
        }
    }

    /// CPU 1D FFT (Cooley-Tukey radix-2) - matches `Pppm::fft_1d_cpu` exactly
    fn fft_1d(data: &mut [f64], n: usize, inverse: bool) {
        use std::f64::consts::PI;

        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                data.swap(i * 2, j * 2);
                data.swap(i * 2 + 1, j * 2 + 1);
            }
            let mut m = n >> 1;
            while m > 0 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey butterfly
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle_step = sign * 2.0 * PI / len as f64;

            for start in (0..n).step_by(len) {
                let mut angle: f64 = 0.0;
                for k in 0..half {
                    let i = start + k;
                    let j_idx = start + k + half;

                    // Twiddle factor
                    let tw_re = angle.cos();
                    let tw_im = angle.sin();

                    // Butterfly
                    let t_re = tw_re * data[j_idx * 2] - tw_im * data[j_idx * 2 + 1];
                    let t_im = tw_re * data[j_idx * 2 + 1] + tw_im * data[j_idx * 2];

                    data[j_idx * 2] = data[i * 2] - t_re;
                    data[j_idx * 2 + 1] = data[i * 2 + 1] - t_im;
                    data[i * 2] += t_re;
                    data[i * 2 + 1] += t_im;

                    angle += angle_step;
                }
            }
            len *= 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fft_roundtrip() {
        // Simple roundtrip test: forward then inverse should return original
        let mesh = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kx = 2;
        let ky = 2;
        let kz = 2;

        let fft_result = PppmCpuFft::forward_3d(&mesh, kx, ky, kz);
        let recovered = PppmCpuFft::inverse_3d(&fft_result, kx, ky, kz);

        for (orig, rec) in mesh.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10, "FFT roundtrip failed");
        }
    }
}
