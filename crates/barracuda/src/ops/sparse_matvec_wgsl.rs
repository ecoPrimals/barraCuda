// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse Matrix-Vector Product (CSR format) - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// GPU shader for f64 sparse matrix-vector product (CSR format).
///
/// Includes `SpMV`, axpy, dot product, scale, copy, diagonal preconditioner,
/// linear combination, and full CG solver kernels — all in f64.
///
/// Entry points: `spmv_f64`, `axpy_f64`, `dot_f64`, `scale_f64`, `copy_f64`,
/// `precond_f64`, `linear_comb_f64`, `final_reduce_f64`, `cg_update_xr`,
/// `cg_update_p`, `compute_alpha`, `compute_beta`.
pub const WGSL_SPARSE_MATVEC_F64: &str = include_str!("../shaders/misc/sparse_matvec_f64.wgsl");

/// Sparse matrix-vector product in CSR (Compressed Sparse Row) format.
pub struct SparseMatVec {
    values: Tensor,
    col_indices: Vec<u32>,
    row_ptrs: Vec<u32>,
    vector: Tensor,
}

impl SparseMatVec {
    /// Create sparse matvec from CSR format (values, `col_indices`, `row_ptrs`) and dense vector.
    #[must_use]
    pub fn new(values: Tensor, col_indices: Vec<u32>, row_ptrs: Vec<u32>, vector: Tensor) -> Self {
        Self {
            values,
            col_indices,
            row_ptrs,
            vector,
        }
    }

    /// Execute sparse matrix-vector product.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.values.device();
        let num_rows = self.row_ptrs.len() - 1;

        if num_rows == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(num_rows)?;

        // Create buffers for CSR data
        let col_indices_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SparseMatVec col_indices"),
                    contents: bytemuck::cast_slice(&self.col_indices),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let row_ptrs_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SparseMatVec row_ptrs"),
                contents: bytemuck::cast_slice(&self.row_ptrs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_rows: u32,
        }

        let params = Params {
            num_rows: num_rows as u32,
        };
        let params_buffer = device.create_uniform_buffer("SparseMatVec Params", &params);

        ComputeDispatch::new(device, "SparseMatVec")
            .shader(include_str!("../shaders/misc/sparse_matvec.wgsl"), "main")
            .storage_read(0, self.values.buffer())
            .storage_read(1, &col_indices_buffer)
            .storage_read(2, &row_ptrs_buffer)
            .storage_read(3, self.vector.buffer())
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch_1d(num_rows as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_rows],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sparse_matvec() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        // values: 1, 2, 3
        // col_indices: 0, 2, 1
        // row_ptrs: 0, 2, 3
        let values = vec![1.0f32, 2.0, 3.0];
        let col_indices = vec![0u32, 2, 1];
        let row_ptrs = vec![0u32, 2, 3];
        let vector = vec![1.0f32, 2.0, 3.0];

        let values_tensor = Tensor::new(values, vec![3], device.clone());
        let vector_tensor = Tensor::new(vector, vec![3], device);

        let output = SparseMatVec::new(values_tensor, col_indices, row_ptrs, vector_tensor)
            .execute()
            .unwrap();

        let result = output.to_vec().unwrap();
        // Row 0: 1*1 + 2*3 = 7
        // Row 1: 3*2 = 6
        assert_eq!(result.len(), 2);
        assert!((result[0] - 7.0).abs() < 1e-5);
        assert!((result[1] - 6.0).abs() < 1e-5);
    }
}
