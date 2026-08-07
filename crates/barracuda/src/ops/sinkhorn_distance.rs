// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SinkhornDistance` - Pure WGSL
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

/// Sinkhorn Distance operation
pub struct SinkhornDistance {
    dist1: Tensor,
    dist2: Tensor,
    cost_matrix: Tensor,
    num_iterations: u32,
    epsilon: f32,
}

impl SinkhornDistance {
    /// Create a new Sinkhorn distance operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        dist1: Tensor,
        dist2: Tensor,
        cost_matrix: Tensor,
        num_iterations: Option<u32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        let dist1_size = dist1.shape().iter().product::<usize>();
        let dist2_size = dist2.shape().iter().product::<usize>();

        if dist1_size != dist2_size {
            return Err(BarracudaError::invalid_op(
                "sinkhorn_distance",
                "dist1 and dist2 must have same size",
            ));
        }

        let size = dist1_size;
        let cost_shape = cost_matrix.shape();
        if cost_shape.len() != 2 || cost_shape[0] != size || cost_shape[1] != size {
            return Err(BarracudaError::invalid_op(
                "sinkhorn_distance",
                "cost_matrix must be [size, size]",
            ));
        }

        Ok(Self {
            dist1,
            dist2,
            cost_matrix,
            num_iterations: num_iterations.unwrap_or(50),
            epsilon: epsilon.unwrap_or(0.1),
        })
    }

    /// Execute the Sinkhorn distance operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.dist1.device();

        let size = self.dist1.shape().iter().product::<usize>();

        // Output is scalar distance
        let output_buffer = device.create_buffer_f32(1)?;

        // Transport plan buffer
        let transport_size = size * size;
        let transport_buffer = device.create_buffer_f32(transport_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            num_iterations: u32,
            epsilon: f32,
            _padding: u32,
        }

        let params = Params {
            size: size as u32,
            num_iterations: self.num_iterations,
            epsilon: self.epsilon,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SinkhornDistance Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "SinkhornDistance")
            .shader(
                include_str!("../shaders/math/sinkhorn_distance_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.dist1.buffer())
            .storage_read(1, self.dist2.buffer())
            .storage_read(2, self.cost_matrix.buffer())
            .storage_rw(3, &transport_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sinkhorn_distance_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 4;

        let dist1 = Tensor::from_vec_on(vec![0.25; size], vec![size], device.clone())
            .await
            .unwrap();

        let dist2 = Tensor::from_vec_on(vec![0.25; size], vec![size], device.clone())
            .await
            .unwrap();

        let cost_matrix =
            Tensor::from_vec_on(vec![1.0; size * size], vec![size, size], device.clone())
                .await
                .unwrap();

        let output = SinkhornDistance::new(dist1, dist2, cost_matrix, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[1]);
    }
}
