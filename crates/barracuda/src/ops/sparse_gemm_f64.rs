// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sparse GEMM (SpMM) — CSR × Dense matrix multiplication on GPU.
//!
//! `C[M, N] = A_csr[M, K] × B_dense[K, N]`
//!
//! Uses one GPU thread per output element (row, col). Each thread iterates
//! over the non-zeros in its CSR row, gathering from the dense B matrix.

use std::sync::Arc;

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;
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
    pub csr: &'a CsrMatrix,
    pub dense_b: &'a [f64],
    pub b_cols: usize,
}

impl SparseGemmF64<'_> {
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

        let shader = device.compile_shader(
            include_str!("../shaders/sparse/spmm_f64.wgsl"),
            Some("spmm_f64"),
        );

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("spmm BGL"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                    uniform_entry(5),
                ],
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spmm BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("spmm PL"),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spmm_f64"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("spmm"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spmm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((output_size as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spmm:staging"),
            size: (output_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &c_buf,
            0,
            &staging,
            0,
            (output_size * std::mem::size_of::<f64>()) as u64,
        );

        device.submit_and_poll(Some(encoder.finish()));

        let result: Vec<f64> = device.map_staging_buffer(&staging, output_size)?;
        Ok(result)
    }

    fn f64_buf(device: &Arc<WgpuDevice>, label: &str, data: &[f64]) -> wgpu::Buffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
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

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;
    use crate::linalg::sparse::CsrMatrix;

    fn spmm_cpu(csr: &CsrMatrix, b: &[f64], n: usize) -> Vec<f64> {
        let m = csr.n_rows;
        let mut c = vec![0.0; m * n];
        for row in 0..m {
            for j in csr.row_ptr[row]..csr.row_ptr[row + 1] {
                let col_a = csr.col_indices[j];
                let val = csr.values[j];
                for col_b in 0..n {
                    c[row * n + col_b] += val * b[col_a * n + col_b];
                }
            }
        }
        c
    }

    #[tokio::test]
    async fn test_spmm_small() {
        let Some(device) = get_test_device_if_gpu_available().await else {
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
        let Some(device) = get_test_device_if_gpu_available().await else {
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
