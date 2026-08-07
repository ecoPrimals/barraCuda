// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GraphNorm` - Pure WGSL
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

/// Graph Normalization operation
pub struct GraphNorm {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_nodes: usize,
    num_features: usize,
    epsilon: f32,
}

impl GraphNorm {
    /// Create a new graph normalization operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, gamma: Tensor, beta: Tensor, epsilon: Option<f32>) -> Result<Self> {
        let input_shape = input.shape();
        let num_nodes = input_shape[0];
        let num_features = input_shape[1..].iter().product::<usize>();

        let gamma_size = gamma.shape().iter().product::<usize>();
        if gamma_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_norm",
                "gamma must have num_features elements",
            ));
        }

        let beta_size = beta.shape().iter().product::<usize>();
        if beta_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_norm",
                "beta must have num_features elements",
            ));
        }

        Ok(Self {
            input,
            gamma,
            beta,
            num_nodes,
            num_features,
            epsilon: epsilon.unwrap_or(1e-5),
        })
    }

    /// Execute the graph normalization operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let output_size = self.num_nodes * self.num_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_features: u32,
            epsilon: f32,
            _padding: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_features: self.num_features as u32,
            epsilon: self.epsilon,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GraphNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "GraphNorm")
            .shader(include_str!("../shaders/norm/graph_norm_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.gamma.buffer())
            .storage_read(2, self.beta.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(self.num_features as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.num_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graph_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 3;
        let num_features = 4;

        let input = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gamma =
            Tensor::from_vec_on(vec![1.0; num_features], vec![num_features], device.clone())
                .await
                .unwrap();

        let beta = Tensor::from_vec_on(vec![0.0; num_features], vec![num_features], device.clone())
            .await
            .unwrap();

        let output = GraphNorm::new(input, gamma, beta, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[num_nodes, num_features]);
    }
}
