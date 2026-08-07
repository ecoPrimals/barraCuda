// SPDX-License-Identifier: AGPL-3.0-or-later
//! Normalize - L2 normalization along dimension (Pure WGSL)
//!
//! Normalizes vectors to unit length: `x_normalized` = x / ||x||_2
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// L2 Normalization
pub struct Normalize {
    input: Tensor,
    dim: usize,
    epsilon: f32,
}

impl Normalize {
    /// Create an L2 normalization operation along the given dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if dim is out of bounds.
    pub fn new(input: Tensor, dim: usize, epsilon: f32) -> Result<Self> {
        let input_shape = input.shape();
        if dim >= input_shape.len() {
            return Err(BarracudaError::invalid_op(
                "normalize",
                "dim must be less than input rank",
            ));
        }

        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "normalize",
                "epsilon must be positive",
            ));
        }

        Ok(Self {
            input,
            dim,
            epsilon,
        })
    }

    /// Execute L2 normalization and return the result tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let dim_size = input_shape[self.dim];

        // Compute outer (product of dimensions before dim)
        let outer = input_shape[..self.dim].iter().product::<usize>();

        // Compute inner (product of dimensions after dim)
        let inner = input_shape[self.dim + 1..].iter().product::<usize>();

        let output_size = input_shape.iter().product::<usize>();
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            outer: u32,
            dim_size: u32,
            inner: u32,
            epsilon: f32,
        }

        let params = Params {
            outer: outer as u32,
            dim_size: dim_size as u32,
            inner: inner as u32,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Normalize Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Normalize")
            .shader(include_str!("../shaders/norm/normalize_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d((outer * inner) as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_normalize_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let normalize = Normalize::new(input, 0, 1e-8).unwrap();
        let output = normalize.execute().unwrap();

        assert_eq!(output.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_normalize_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 12], vec![3, 4], device.clone())
            .await
            .unwrap();

        let normalize = Normalize::new(input, 1, 1e-8).unwrap();
        let output = normalize.execute().unwrap();

        assert_eq!(output.shape(), &[3, 4]);
    }
}
