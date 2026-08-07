// SPDX-License-Identifier: AGPL-3.0-or-later
//! Outer Product - Parallel outer product (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Parallel outer product
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates vector inputs
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OuterProductParams {
    size_a: u32,
    size_b: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Outer product of two 1D vectors producing a 2D matrix.
pub struct OuterProduct {
    vec_a: Tensor,
    vec_b: Tensor,
}

impl OuterProduct {
    /// Creates a new outer product. Both inputs must be 1D vectors.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(vec_a: Tensor, vec_b: Tensor) -> Result<Self> {
        if vec_a.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "outer_product",
                "First input must be 1D vector",
            ));
        }

        if vec_b.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "outer_product",
                "Second input must be 1D vector",
            ));
        }

        Ok(Self { vec_a, vec_b })
    }

    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/misc/outer_product_f64.wgsl");
            SHADER
        }
    }

    /// Executes the outer product and returns the result matrix.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.vec_a.device();
        let size_a = self.vec_a.len();
        let size_b = self.vec_b.len();
        let output_size = size_a * size_b;

        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = OuterProductParams {
            size_a: size_a as u32,
            size_b: size_b as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("OuterProduct Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (size_b as u32).div_ceil(16);
        let workgroups_y = (size_a as u32).div_ceil(16);

        ComputeDispatch::new(device, "OuterProduct")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.vec_a.buffer())
            .storage_read(2, self.vec_b.buffer())
            .storage_rw(3, &output_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![size_a, size_b],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_outer_product_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![4.0, 5.0], vec![2], device.clone())
            .await
            .unwrap();

        let result = OuterProduct::new(a, b).unwrap().execute().unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        let output = result.to_vec().unwrap();
        assert_eq!(output[0], 4.0); // 1*4
        assert_eq!(output[1], 5.0); // 1*5
        assert_eq!(output[2], 8.0); // 2*4
    }
}
