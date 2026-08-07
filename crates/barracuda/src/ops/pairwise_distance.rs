// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PairwiseDistance` - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Pairwise Distance operation
pub struct PairwiseDistance {
    input1: Tensor,
    input2: Tensor,
    p: f32,
    epsilon: f32,
}

impl PairwiseDistance {
    /// Create a new pairwise distance operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input1: Tensor,
        input2: Tensor,
        p: Option<f32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        let input1_shape = input1.shape();
        let input2_shape = input2.shape();

        if input1_shape.len() < 2 || input2_shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "pairwise_distance",
                "inputs must be at least 2D",
            ));
        }

        let num_pairs = input1_shape[0];
        let dim = input1_shape[1..].iter().product::<usize>();

        if input2_shape[0] != num_pairs || input2_shape[1..].iter().product::<usize>() != dim {
            return Err(BarracudaError::invalid_op(
                "pairwise_distance",
                "input shapes must match",
            ));
        }

        Ok(Self {
            input1,
            input2,
            p: p.unwrap_or(2.0),
            epsilon: epsilon.unwrap_or(1e-8),
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        const SHADER: &str = include_str!("../shaders/math/pairwise_distance_f64.wgsl");
        SHADER
    }

    /// Execute the pairwise distance operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input1.device();

        let num_pairs = self.input1.shape()[0];
        let dim = self.input1.shape()[1..].iter().product::<usize>();
        let output_size = num_pairs;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_pairs: u32,
            dim: u32,
            p: f32,
            epsilon: f32,
        }

        let params = Params {
            num_pairs: num_pairs as u32,
            dim: dim as u32,
            p: self.p,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PairwiseDistance Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "PairwiseDistance")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input1.buffer())
            .storage_read(1, self.input2.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(num_pairs as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_pairs],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pairwise_distance_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_pairs = 3;
        let dim = 2;

        let input1 = Tensor::from_vec_on(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![num_pairs, dim],
            device.clone(),
        )
        .await
        .unwrap();

        let input2 = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![num_pairs, dim],
            device.clone(),
        )
        .await
        .unwrap();

        let output = PairwiseDistance::new(input1, input2, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[num_pairs]);
    }
}
