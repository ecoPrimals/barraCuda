// SPDX-License-Identifier: AGPL-3.0-or-later
//! Repeat Interleave - Pure WGSL
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

/// Repeat interleave operation
pub struct RepeatInterleave {
    input: Tensor,
    repeats: usize,
    dim: usize,
}

impl RepeatInterleave {
    /// Create a new repeat interleave operation
    /// # Errors
    /// Returns [`Err`] if `dim` exceeds the tensor rank or if `repeats` is zero.
    pub fn new(input: Tensor, repeats: usize, dim: usize) -> Result<Self> {
        let shape = input.shape();
        if dim >= shape.len() {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "dim {} exceeds tensor rank {}",
                dim,
                shape.len()
            )));
        }

        if repeats == 0 {
            return Err(crate::error::BarracudaError::invalid_input(
                "repeats must be positive",
            ));
        }

        Ok(Self {
            input,
            repeats,
            dim,
        })
    }

    /// Get the WGSL shader source
    /// Execute the repeat interleave operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let dim_size = shape[self.dim];

        // Compute output shape
        let mut output_shape = shape.to_vec();
        output_shape[self.dim] = dim_size * self.repeats;
        let output_size: usize = output_shape.iter().product();
        let input_size: usize = shape.iter().product();

        // Compute inner and outer sizes
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let outer_size: usize = shape[..self.dim].iter().product();

        // Access input buffer directly (zero-copy)
        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            input_size: u32,
            repeats: u32,
            dim: u32,
            dim_size: u32,
            inner_size: u32,
            outer_size: u32,
            _pad1: u32,
        }

        let params = Params {
            output_size: output_size as u32,
            input_size: input_size as u32,
            repeats: self.repeats as u32,
            dim: self.dim as u32,
            dim_size: dim_size as u32,
            inner_size: inner_size as u32,
            outer_size: outer_size as u32,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RepeatInterleave Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "RepeatInterleave")
            .shader(
                include_str!("../shaders/tensor/repeat_interleave_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_repeat_interleave_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let result = RepeatInterleave::new(input, 2, 0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![6]);
    }

    #[tokio::test]
    async fn test_repeat_interleave_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3], device).unwrap();

        let result = RepeatInterleave::new(input, 3, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![2, 9]);
    }

    #[tokio::test]
    async fn test_repeat_interleave_invalid_dim() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(RepeatInterleave::new(input, 2, 10).is_err());
    }

    #[tokio::test]
    async fn test_repeat_interleave_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(RepeatInterleave::new(input, 0, 0).is_err());
    }

    #[tokio::test]
    async fn test_repeat_interleave_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![10, 10], device).unwrap();

        let result = RepeatInterleave::new(input, 5, 0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(result.shape(), &vec![50, 10]);
    }
}
