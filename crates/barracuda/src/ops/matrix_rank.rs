// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matrix Rank - Compute rank of matrix (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Gaussian elimination on GPU
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates matrix dimensions
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/matrix_rank_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
const SHADER_F32: &str = SHADER_F64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatrixRankParams {
    rows: u32,
    cols: u32,
    tolerance: f32,
    _pad1: u32,
}

/// Matrix rank computation via Gaussian elimination on GPU.
pub struct MatrixRank {
    input: Tensor,
    tolerance: f32,
}

impl MatrixRank {
    /// Creates a new matrix rank operation. Tolerance controls numerical rank threshold.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, tolerance: f32) -> Result<Self> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "matrix_rank",
                "Requires at least 2D tensor",
            ));
        }

        Ok(Self { input, tolerance })
    }

    fn wgsl_shader() -> &'static str {
        SHADER_F32
    }

    /// Executes rank computation and returns the matrix rank.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<usize> {
        let device = self.input.device();
        let shape = self.input.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let total_elements = rows * cols;

        // Create work matrix buffer
        let work_matrix_buffer = device.create_buffer_f32(total_elements)?;

        // Create rank output buffer (single u32)
        let rank_buffer = device.create_buffer_u32(1)?;

        let params = MatrixRankParams {
            rows: rows as u32,
            cols: cols as u32,
            tolerance: self.tolerance,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatrixRank Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shader_src = Self::wgsl_shader();
        let min_dim = rows.min(cols);
        let mut batch = BatchedComputeDispatch::new(device);

        batch.push(
            ComputeDispatch::new(device, "MatrixRank Copy")
                .shader(shader_src, "copy_matrix")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_rw(2, &work_matrix_buffer)
                .storage_rw(3, &rank_buffer)
                .dispatch_1d(total_elements as u32),
        )?;

        batch.push(
            ComputeDispatch::new(device, "MatrixRank Gaussian")
                .shader(shader_src, "gaussian_elimination")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_rw(2, &work_matrix_buffer)
                .storage_rw(3, &rank_buffer)
                .dispatch(min_dim as u32, 1, 1),
        )?;

        batch.push(
            ComputeDispatch::new(device, "MatrixRank Count")
                .shader(shader_src, "count_rank")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_rw(2, &work_matrix_buffer)
                .storage_rw(3, &rank_buffer)
                .dispatch_1d(rows as u32),
        )?;

        batch.submit()?;

        // Read rank result
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MatrixRank Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut read_encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("MatrixRank Read Encoder"),
        });
        read_encoder.copy_buffer_to_buffer(
            &rank_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<u32>() as u64,
        );
        device.submit_commands(Some(read_encoder.finish()));

        let rank_data: Vec<u32> = device.map_staging_buffer(&staging_buffer, 1)?;
        Ok(rank_data[0] as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_matrix_rank_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![1.0, 2.0, 2.0, 4.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let rank = MatrixRank::new(matrix, 1e-6).unwrap().execute().unwrap();
        assert_eq!(rank, 1); // Rank 1 (second row is 2x first)
    }

    #[tokio::test]
    async fn test_matrix_rank_full_rank() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let rank = MatrixRank::new(matrix, 1e-6).unwrap().execute().unwrap();
        assert_eq!(rank, 2);
    }

    #[tokio::test]
    async fn test_matrix_rank_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let rank = MatrixRank::new(matrix, 1e-6).unwrap().execute().unwrap();
        assert_eq!(rank, 0);
    }
}
