// SPDX-License-Identifier: AGPL-3.0-or-later
//! 2D Fast Fourier Transform Operation
//!
//! **Purpose**: 2D frequency analysis for images and 2D signals
//! **Algorithm**: Row-column decomposition using 1D FFT

use super::Fft1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// 2D Complex FFT operation
pub struct Fft2D {
    input: Tensor,
    rows: u32,
    cols: u32,
}

impl Fft2D {
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

        if rows & (rows - 1) != 0 || cols & (cols - 1) != 0 {
            return Err(BarracudaError::Device(
                "FFT 2D dimensions must be powers of 2".to_string(),
            ));
        }

        Ok(Self { input, rows, cols })
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Step 1: FFT each row
        let mut row_results = Vec::new();

        for row_idx in 0..self.rows {
            let row_data = self.extract_row(row_idx)?;
            let row_tensor =
                Tensor::from_data(&row_data, vec![self.cols as usize, 2], device.clone())?;
            let fft = Fft1D::new(row_tensor, self.cols)?;
            let row_fft = fft.execute()?;
            row_results.extend(row_fft.to_vec()?);
        }

        let rows_transformed = Tensor::from_data(
            &row_results,
            vec![self.rows as usize, self.cols as usize, 2],
            device.clone(),
        )?;

        // Step 2: FFT each column (via transpose)
        let transposed_data = self.transpose_2d_complex(&rows_transformed)?;
        let mut col_results = Vec::new();

        for col_idx in 0..self.cols {
            let col_data = self.extract_from_transposed(&transposed_data, col_idx)?;
            let col_tensor =
                Tensor::from_data(&col_data, vec![self.rows as usize, 2], device.clone())?;
            let fft = Fft1D::new(col_tensor, self.rows)?;
            col_results.extend(fft.execute()?.to_vec()?);
        }

        let cols_transformed_flat: Vec<f32> = col_results;
        let final_data = self.transpose_flat_2d_complex(&cols_transformed_flat)?;

        Tensor::from_data(
            &final_data,
            vec![self.rows as usize, self.cols as usize, 2],
            device.clone(),
        )
    }

    fn extract_row(&self, row_idx: u32) -> Result<Vec<f32>> {
        let data = self.input.to_vec()?;
        let cols = self.cols as usize;
        let start = (row_idx as usize) * cols * 2;
        let end = start + cols * 2;
        Ok(data[start..end].to_vec())
    }

    fn extract_from_transposed(&self, transposed: &[f32], row_idx: u32) -> Result<Vec<f32>> {
        let rows = self.rows as usize;
        let start = (row_idx as usize) * rows * 2;
        let end = start + rows * 2;
        Ok(transposed[start..end].to_vec())
    }

    fn transpose_2d_complex(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        let data = tensor.to_vec()?;
        let m = self.rows as usize;
        let n = self.cols as usize;
        let mut transposed = vec![0.0f32; m * n * 2];

        for i in 0..m {
            for j in 0..n {
                let src_base = (i * n + j) * 2;
                let dst_base = (j * m + i) * 2;
                transposed[dst_base] = data[src_base];
                transposed[dst_base + 1] = data[src_base + 1];
            }
        }

        Ok(transposed)
    }

    fn transpose_flat_2d_complex(&self, data: &[f32]) -> Result<Vec<f32>> {
        let m = self.cols as usize; // After first transpose: cols×rows
        let n = self.rows as usize;
        let mut transposed = vec![0.0f32; m * n * 2];

        for i in 0..m {
            for j in 0..n {
                let src_base = (i * n + j) * 2;
                let dst_base = (j * m + i) * 2;
                transposed[dst_base] = data[src_base];
                transposed[dst_base + 1] = data[src_base + 1];
            }
        }

        Ok(transposed)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
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
