// SPDX-License-Identifier: AGPL-3.0-or-later
//! 2D Fast Fourier Transform Operation
//!
//! **Purpose**: 2D frequency analysis for images and 2D signals
//! **Algorithm**: Row-column decomposition using batched 1D FFT (GPU-resident, zero CPU readbacks)

use super::fft_1d::{batched_shader_f32, upload_twiddles_f32, AxisConfig};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::mem::size_of;

/// 2D Complex FFT operation
pub struct Fft2D {
    input: Tensor,
    rows: u32,
    cols: u32,
}

impl Fft2D {
    /// Create 2D FFT for [rows, cols, 2] complex input (powers of 2).
    /// # Errors
    /// Returns [`Err`] if input is not 3D [rows, cols, 2], or dimensions are not powers of 2.
    pub fn new(input: Tensor, rows: u32, cols: u32) -> Result<Self> {
        let shape = input.shape();

        if shape.len() != 3 {
            return Err(BarracudaError::Device(
                "FFT 2D input must have 3 dimensions [rows, cols, 2]".to_string(),
            ));
        }

        if shape[0] != rows as usize || shape[1] != cols as usize || shape[2] != 2 {
            return Err(BarracudaError::Device(format!(
                "FFT 2D shape mismatch: expected [{rows}, {cols}, 2], got {shape:?}"
            )));
        }

        if rows == 0 || cols == 0 {
            return Err(BarracudaError::Device(
                "FFT 2D dimensions must be non-zero".to_string(),
            ));
        }

        if rows & (rows - 1) != 0 || cols & (cols - 1) != 0 {
            return Err(BarracudaError::Device(
                "FFT 2D dimensions must be powers of 2".to_string(),
            ));
        }

        Ok(Self { input, rows, cols })
    }

    /// Execute 2D FFT (row-wise then column-wise). Zero CPU readbacks.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let buffer_bytes = (self.rows as u64) * (self.cols as u64) * 2 * size_of::<f32>() as u64;

        // buf_a holds data, buf_b is ping-pong
        let buf_a = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT2D buf_a"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT2D Copy Input"),
            });
            encoder.copy_buffer_to_buffer(
                self.input.buffer(),
                0,
                &buf_a,
                0,
                buffer_bytes,
            );
            device.submit_and_poll(std::iter::once(encoder.finish()));
        }

        let buf_b = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT2D buf_b"),
            size: buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = batched_shader_f32();

        // Row pass: degree=cols, element_stride=1, dim1_stride=cols, dim1_count=rows, dim2_count=1
        let tw_row_re;
        let tw_row_im;
        {
            let (re, im) = upload_twiddles_f32(device, self.cols);
            tw_row_re = re;
            tw_row_im = im;
        }
        let row_axis = AxisConfig {
            degree: self.cols,
            element_stride: 1,
            dim1_stride: self.cols,
            dim2_stride: 1,
            dim1_count: self.rows,
            dim2_count: 1,
        };
        super::fft_1d::dispatch_axis_inner(
            device,
            shader,
            &buf_a,
            &buf_b,
            buffer_bytes,
            &tw_row_re,
            &tw_row_im,
            &row_axis,
            false,
            false,
        )?;

        // Col pass: degree=rows, element_stride=cols, dim1_stride=1, dim1_count=cols, dim2_count=1
        let tw_col_re;
        let tw_col_im;
        {
            let (re, im) = upload_twiddles_f32(device, self.rows);
            tw_col_re = re;
            tw_col_im = im;
        }
        let col_axis = AxisConfig {
            degree: self.rows,
            element_stride: self.cols,
            dim1_stride: 1,
            dim2_stride: 1,
            dim1_count: self.cols,
            dim2_count: 1,
        };
        super::fft_1d::dispatch_axis_inner(
            device,
            shader,
            &buf_a,
            &buf_b,
            buffer_bytes,
            &tw_col_re,
            &tw_col_im,
            &col_axis,
            false,
            false,
        )?;

        Ok(Tensor::from_buffer(
            buf_a,
            vec![self.rows as usize, self.cols as usize, 2],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fft_2d_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let data = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        let tensor = Tensor::from_data(&data, vec![2, 2, 2], device.clone()).unwrap();
        let fft = Fft2D::new(tensor, 2, 2).unwrap();
        let result = fft.execute().unwrap();
        let result_data = result.to_vec().unwrap();
        assert_eq!(result_data.len(), 8);
    }
}
