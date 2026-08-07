// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pdist - Pure WGSL
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

/// Pairwise Distance (all pairs) operation
pub struct Pdist {
    input: Tensor,
    p: f32,
    epsilon: f32,
}

impl Pdist {
    /// Create a new pdist operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, p: Option<f32>, epsilon: Option<f32>) -> Result<Self> {
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "pdist",
                "input must be at least 2D",
            ));
        }

        Ok(Self {
            input,
            p: p.unwrap_or(2.0),
            epsilon: epsilon.unwrap_or(1e-8),
        })
    }

    /// Get the WGSL shader source
    /// Execute the pdist operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let input_shape = self.input.shape();
        let num_vectors = input_shape[0];
        let dim = input_shape[1..].iter().product::<usize>();

        // Output is condensed distance matrix: n*(n-1)/2 pairs
        let num_pairs = num_vectors * (num_vectors - 1) / 2;
        let output_buffer = device.create_buffer_f32(num_pairs)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_vectors: u32,
            dim: u32,
            p: f32,
            epsilon: f32,
        }

        let params = Params {
            num_vectors: num_vectors as u32,
            dim: dim as u32,
            p: self.p,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Pdist Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (num_vectors as u32).div_ceil(256);
        let workgroups_y = (num_vectors as u32).div_ceil(256);

        ComputeDispatch::new(device, "Pdist")
            .shader(include_str!("../shaders/misc/pdist_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
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
    async fn test_pdist_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_vectors = 3;
        let dim = 2;

        let input = Tensor::from_vec_on(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            vec![num_vectors, dim],
            device.clone(),
        )
        .await
        .unwrap();

        let output = Pdist::new(input, None, None).unwrap().execute().unwrap();

        // 3 choose 2 = 3 pairs
        assert_eq!(output.shape(), &[3]);
    }
}
