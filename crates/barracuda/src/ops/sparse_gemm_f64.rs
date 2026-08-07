// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sparse GEMM (`SpMM`) — CSR × Dense matrix multiplication on GPU.
//!
//! `C[M, N] = A_csr[M, K] × B_dense[K, N]`
//!
//! Uses one GPU thread per output element (row, col). Each thread iterates
//! over the non-zeros in its CSR row, gathering from the dense B matrix.

use std::sync::Arc;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::linalg::sparse::CsrMatrix;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmmParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

/// Sparse matrix × dense matrix product on GPU.
///
/// Computes `C = A × B` where `A` is CSR `[M, K]` and `B` is dense `[K, N]`.
/// Returns `C` as a flat `Vec<f64>` in row-major order `[M, N]`.
pub struct SparseGemmF64<'a> {
    /// CSR sparse matrix [M, K].
    pub csr: &'a CsrMatrix,
    /// Dense matrix B [K, N] in row-major order.
    pub dense_b: &'a [f64],
    /// Number of columns in B (N).
    pub b_cols: usize,
}

impl SparseGemmF64<'_> {
    /// Execute sparse-dense matrix multiplication on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(&self, device: &Arc<WgpuDevice>) -> Result<Vec<f64>> {
        let m = self.csr.n_rows;
        let k = self.csr.n_cols;
        let n = self.b_cols;

        if self.dense_b.len() != k * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![k, n],
                actual: vec![self.dense_b.len()],
            });
        }

        let nnz = self.csr.values.len();
        if nnz == 0 {
            return Ok(vec![0.0; m * n]);
        }

        let values_buf = Self::f64_buf(device, "spmm:values", &self.csr.values);
        let col_indices: Vec<u32> = self.csr.col_indices.iter().map(|&c| c as u32).collect();
        let col_buf = Self::u32_buf(device, "spmm:col_idx", &col_indices);
        let row_ptr: Vec<u32> = self.csr.row_ptr.iter().map(|&r| r as u32).collect();
        let row_buf = Self::u32_buf(device, "spmm:row_ptr", &row_ptr);
        let b_buf = Self::f64_buf(device, "spmm:B", self.dense_b);

        let output_size = m * n;
        let c_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spmm:C"),
            size: (output_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = SpmmParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spmm:params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "spmm_f64")
            .shader(
                include_str!("../shaders/sparse/spmm_f64.wgsl"),
                "main",
            )
            .f64()
            .storage_read(0, &values_buf)
            .storage_read(1, &col_buf)
            .storage_read(2, &row_buf)
            .storage_read(3, &b_buf)
            .storage_rw(4, &c_buf)
            .uniform(5, &params_buf)
            .dispatch_1d(output_size as u32)
            .submit()?;

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spmm:staging"),
            size: (output_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("spmm readback"),
        });
        encoder.copy_buffer_to_buffer(
            &c_buf,
            0,
            &staging,
            0,
            (output_size * std::mem::size_of::<f64>()) as u64,
        );

        device.submit_commands(Some(encoder.finish()));

        let result: Vec<f64> = device.map_staging_buffer(&staging, output_size)?;
        Ok(result)
    }

    fn f64_buf(device: &Arc<WgpuDevice>, label: &str, data: &[f64]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn u32_buf(device: &Arc<WgpuDevice>, label: &str, data: &[u32]) -> wgpu::Buffer {
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;
    use crate::linalg::sparse::CsrMatrix;

    fn spmm_cpu(csr: &CsrMatrix, b: &[f64], n: usize) -> Vec<f64> {
        let m = csr.n_rows;
        let mut c = vec![0.0; m * n];
        for row in 0..m {
            for j in csr.row_ptr[row]..csr.row_ptr[row + 1] {
                let col_a = csr.col_indices[j];
                let val = csr.values[j];
                for col_b in 0..n {
                    c[row * n + col_b] = val.mul_add(b[col_a * n + col_b], c[row * n + col_b]);
                }
            }
        }
        c
    }

    #[tokio::test]
    async fn test_spmm_small() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        // 3×4 CSR × 4×2 dense
        let csr = CsrMatrix {
            n_rows: 3,
            n_cols: 4,
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            col_indices: vec![0, 2, 1, 3, 0],
            row_ptr: vec![0, 2, 4, 5],
        };
        let b = vec![
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
            7.0, 8.0, // row 3
        ];
        let expected = spmm_cpu(&csr, &b, 2);
        let op = SparseGemmF64 {
            csr: &csr,
            dense_b: &b,
            b_cols: 2,
        };
        let got = op.execute(&device).unwrap();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-10,
                "mismatch at {i}: got {g}, expected {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_spmm_identity() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        // 4×4 identity × 4×3 dense = dense
        let csr = CsrMatrix {
            n_rows: 4,
            n_cols: 4,
            values: vec![1.0; 4],
            col_indices: vec![0, 1, 2, 3],
            row_ptr: vec![0, 1, 2, 3, 4],
        };
        let b: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let op = SparseGemmF64 {
            csr: &csr,
            dense_b: &b,
            b_cols: 3,
        };
        let got = op.execute(&device).unwrap();
        for (i, (g, e)) in got.iter().zip(b.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-10,
                "identity mismatch at {i}: got {g}, expected {e}"
            );
        }
    }
}
