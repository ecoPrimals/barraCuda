// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triu - Complete triangular (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: OVERWRITE existing CPU version
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates matrix dimensions
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/triu_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TriuParams {
    rows: u32,
    cols: u32,
    diagonal: i32,
    _pad1: u32,
}

/// Upper triangular matrix extraction operation.
pub struct Triu {
    input: Tensor,
    diagonal: i32,
}

impl Triu {
    /// Creates a new triu operation. `diagonal` selects which diagonal to include (0 = main).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, diagonal: i32) -> Result<Self> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "triu",
                "Requires at least 2D tensor",
            ));
        }

        Ok(Self { input, diagonal })
    }

    /// Executes the triu operation and returns the upper triangular result.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let matrix_size = rows * cols;

        let output_buffer = device.create_buffer_f32(matrix_size)?;

        let params = TriuParams {
            rows: rows as u32,
            cols: cols as u32,
            diagonal: self.diagonal,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Triu Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (cols as u32).div_ceil(16);
        let workgroups_y = (rows as u32).div_ceil(16);

        ComputeDispatch::new(device, "Triu")
            .shader(SHADER_F64, "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_triu_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let result = Triu::new(matrix, 0).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[3], 0.0); // Below diagonal
        assert_eq!(output[4], 5.0);
    }
}
