// SPDX-License-Identifier: AGPL-3.0-or-later
//! Graph Batch Normalization - Batch normalization adapted for graph data (Pure WGSL)
//!
//! Normalizes node features across the batch and feature dimensions
//! Similar to standard batch norm, but operates on graph nodes
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Graph Batch Normalization
pub struct GraphBatchNorm {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    num_nodes: usize,
    num_features: usize,
    epsilon: f32,
}

impl GraphBatchNorm {
    /// Create graph batch normalization.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, gamma: Tensor, beta: Tensor, epsilon: f32) -> Result<Self> {
        let input_shape = input.shape();
        if input_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "input must be 2D [num_nodes, num_features]",
            ));
        }

        let num_nodes = input_shape[0];
        let num_features = input_shape[1];

        let gamma_shape = gamma.shape();
        let gamma_size = gamma_shape.iter().product::<usize>();
        if gamma_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "gamma must have num_features elements",
            ));
        }

        let beta_shape = beta.shape();
        let beta_size = beta_shape.iter().product::<usize>();
        if beta_size != num_features {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "beta must have num_features elements",
            ));
        }

        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "graph_batch_norm",
                "epsilon must be positive",
            ));
        }

        Ok(Self {
            input,
            gamma,
            beta,
            num_nodes,
            num_features,
            epsilon,
        })
    }

    /// Execute graph batch normalization.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        // Create intermediate buffers for mean and variance
        let mean_buffer = device.create_buffer_f32(self.num_features)?;
        let variance_buffer = device.create_buffer_f32(self.num_features)?;

        // Create output buffer
        let output_size = self.num_nodes * self.num_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_features: u32,
            epsilon: f32,
            _pad1: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_features: self.num_features as u32,
            epsilon: self.epsilon,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GraphBatchNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_source = include_str!("../shaders/norm/graph_batch_norm_f64.wgsl");
        let mut batch = BatchedComputeDispatch::new(device);
        batch.push(
            ComputeDispatch::new(device, "GraphBatchNorm Mean")
                .shader(shader_source, "compute_mean")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_read(2, self.gamma.buffer())
                .storage_read(3, self.beta.buffer())
                .storage_rw(4, &output_buffer)
                .storage_rw(5, &mean_buffer)
                .storage_rw(6, &variance_buffer)
                .dispatch_1d(self.num_features as u32),
        )?;
        batch.push(
            ComputeDispatch::new(device, "GraphBatchNorm Variance")
                .shader(shader_source, "compute_variance")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_read(2, self.gamma.buffer())
                .storage_read(3, self.beta.buffer())
                .storage_rw(4, &output_buffer)
                .storage_rw(5, &mean_buffer)
                .storage_rw(6, &variance_buffer)
                .dispatch_1d(self.num_features as u32),
        )?;
        batch.push(
            ComputeDispatch::new(device, "GraphBatchNorm Normalize")
                .shader(shader_source, "normalize")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_read(2, self.gamma.buffer())
                .storage_read(3, self.beta.buffer())
                .storage_rw(4, &output_buffer)
                .storage_rw(5, &mean_buffer)
                .storage_rw(6, &variance_buffer)
                .dispatch_1d(output_size as u32),
        )?;
        batch.submit()?;

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
    async fn test_graph_batch_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 4;
        let num_features = 8;

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

        let batch_norm = GraphBatchNorm::new(input, gamma, beta, 1e-5).unwrap();
        let output = batch_norm.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, num_features]);
    }

    #[tokio::test]
    async fn test_graph_batch_norm_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 100;
        let num_features = 128;

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

        let batch_norm = GraphBatchNorm::new(input, gamma, beta, 1e-5).unwrap();
        let output = batch_norm.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, num_features]);
    }
}
