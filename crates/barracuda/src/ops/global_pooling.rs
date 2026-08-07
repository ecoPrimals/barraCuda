// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Pooling - Graph-level representation aggregation (Pure WGSL)
//!
//! Aggregate node features to graph-level representation
//! Supports: sum, mean, max aggregation
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Aggregation type for global pooling.
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    /// Sum over nodes.
    Sum,
    /// Mean over nodes.
    Mean,
    /// Max over nodes.
    Max,
}

/// Global pooling operation (graph-level aggregation).
pub struct GlobalPooling {
    node_features: Tensor,
    num_nodes: usize,
    num_features: usize,
    aggregation_type: AggregationType,
}

impl GlobalPooling {
    /// Create global pooling with the given aggregation type.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(node_features: Tensor, aggregation_type: AggregationType) -> Result<Self> {
        let node_shape = node_features.shape();
        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "global_pooling",
                "node_features must be 2D [num_nodes, num_features]",
            ));
        }

        let num_nodes = node_shape[0];
        let num_features = node_shape[1];

        Ok(Self {
            node_features,
            num_nodes,
            num_features,
            aggregation_type,
        })
    }

    /// Execute global pooling and return the output tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        // Create output buffer
        let output_buffer = device.create_buffer_f32(self.num_features)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_features: u32,
            aggregation_type: u32,
            _pad1: u32,
        }

        let aggregation_code = match self.aggregation_type {
            AggregationType::Sum => 0u32,
            AggregationType::Mean => 1u32,
            AggregationType::Max => 2u32,
        };

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_features: self.num_features as u32,
            aggregation_type: aggregation_code,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GlobalPooling Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "GlobalPooling")
            .shader(
                include_str!("../shaders/pooling/global_pooling_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, self.node_features.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d(self.num_features as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_global_pooling_sum() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 4;
        let num_features = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Sum).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_mean() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 3;
        let num_features = 4;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Mean).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_max() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 5;
        let num_features = 16;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Max).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }

    #[tokio::test]
    async fn test_global_pooling_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 100;
        let num_features = 128;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * num_features],
            vec![num_nodes, num_features],
            device.clone(),
        )
        .await
        .unwrap();

        let pooling = GlobalPooling::new(node_features, AggregationType::Mean).unwrap();
        let output = pooling.execute().unwrap();

        assert_eq!(output.shape(), &[num_features]);
    }
}
