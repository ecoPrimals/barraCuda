// SPDX-License-Identifier: AGPL-3.0-only
//! 3D Fast Fourier Transform Operation
//!
//! **Purpose**: 3D frequency analysis for PPPM molecular dynamics
//! **Algorithm**: Dimension-wise decomposition using batched 1D FFT (Z, Y, X)
//! **CRITICAL FOR PPPM**: This operation unblocks molecular dynamics!
//!
//! GPU-resident: zero CPU readbacks. Uses `HashMap` for twiddle caching by degree.

use super::fft_1d::{AxisConfig, batched_shader_f32, dispatch_axis_inner, upload_twiddles_f32};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::mem::size_of;

/// 3D Complex FFT operation
pub struct Fft3D {
    input: Tensor,
    nx: u32,
    ny: u32,
    nz: u32,
    twiddles: HashMap<u32, (wgpu::Buffer, wgpu::Buffer)>,
}

impl Fft3D {
    /// Create 3D FFT for [nx, ny, nz, 2] complex input (powers of 2).
    /// # Errors
    /// Returns [`Err`] if input is not 4D [nx, ny, nz, 2], or dimensions are not powers of 2.
    pub fn new(input: Tensor, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(BarracudaError::Device(
                "FFT 3D input must have 4 dimensions [nx, ny, nz, 2]".to_string(),
            ));
        }

        if shape[0] != nx as usize
            || shape[1] != ny as usize
            || shape[2] != nz as usize
            || shape[3] != 2
        {
            return Err(BarracudaError::Device(format!(
                "FFT 3D shape mismatch: expected [{nx}, {ny}, {nz}, 2], got {shape:?}"
            )));
        }

        if nx & (nx - 1) != 0 || ny & (ny - 1) != 0 || nz & (nz - 1) != 0 {
            return Err(BarracudaError::Device(
                "FFT 3D dimensions must be powers of 2".to_string(),
            ));
        }

        let device = input.device();
        let mut twiddles = HashMap::new();
        for &n in &[nx, ny, nz] {
            twiddles
                .entry(n)
                .or_insert_with(|| upload_twiddles_f32(device, n));
        }

        Ok(Self {
            input,
            nx,
            ny,
            nz,
            twiddles,
        })
    }

    /// Execute 3D FFT (Z, Y, X passes). Zero CPU readbacks.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let buffer_bytes =
            (self.nx as u64) * (self.ny as u64) * (self.nz as u64) * 2 * size_of::<f32>() as u64;

        let buf_a = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT3D buf_a"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT3D Copy Input"),
            });
            encoder.copy_buffer_to_buffer(self.input.buffer(), 0, &buf_a, 0, buffer_bytes);
            device.submit_commands(std::iter::once(encoder.finish()));
        }

        let buf_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT3D buf_b"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = batched_shader_f32();

        // Axis configs matching Fft3DF64 pattern (Z, Y, X)
        let axes = [
            AxisConfig {
                degree: self.nz,
                element_stride: 1,
                dim1_stride: self.ny * self.nz,
                dim2_stride: self.nz,
                dim1_count: self.nx,
                dim2_count: self.ny,
            },
            AxisConfig {
                degree: self.ny,
                element_stride: self.nz,
                dim1_stride: self.ny * self.nz,
                dim2_stride: 1,
                dim1_count: self.nx,
                dim2_count: self.nz,
            },
            AxisConfig {
                degree: self.nx,
                element_stride: self.ny * self.nz,
                dim1_stride: self.nz,
                dim2_stride: 1,
                dim1_count: self.ny,
                dim2_count: self.nz,
            },
        ];

        for axis in &axes {
            let (tw_re, tw_im) = &self.twiddles[&axis.degree];
            dispatch_axis_inner(
                device,
                shader,
                &buf_a,
                &buf_b,
                buffer_bytes,
                tw_re,
                tw_im,
                axis,
                false,
                false,
            )?;
        }

        Ok(Tensor::from_buffer(
            buf_a,
            vec![self.nx as usize, self.ny as usize, self.nz as usize, 2],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fft_3d_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let data = vec![
            1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0,
        ];

        let tensor = Tensor::from_data(&data, vec![2, 2, 2, 2], device).unwrap();
        let fft = Fft3D::new(tensor, 2, 2, 2).unwrap();
        let result = fft.execute().unwrap();
        let result_data = result.to_vec().unwrap();
        assert_eq!(result_data.len(), 16);
    }
}
