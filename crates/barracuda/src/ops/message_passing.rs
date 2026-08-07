// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MessagePassing` - Pure WGSL
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

/// Message Passing operation
pub struct MessagePassing {
    node_features: Tensor,
    edge_index: Tensor,
    edge_features: Option<Tensor>,
    num_nodes: usize,
    num_edges: usize,
    node_feat_dim: usize,
    edge_feat_dim: usize,
    message_dim: usize,
    aggr_type: u32, // 0 = sum, 1 = mean, 2 = max
}

impl MessagePassing {
    /// Create a new message passing operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `edge_index` is not [`num_edges`, 2].
    pub fn new(
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Option<Tensor>,
        aggr_type: u32,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        let num_nodes = node_shape[0];
        let node_feat_dim = node_shape[1..].iter().product::<usize>();

        let edge_shape = edge_index.shape();
        if edge_shape.len() != 2 || edge_shape[1] != 2 {
            return Err(BarracudaError::invalid_op(
                "message_passing",
                "edge_index must be [num_edges, 2]",
            ));
        }
        let num_edges = edge_shape[0];

        let edge_feat_dim = if let Some(ref ef) = edge_features {
            ef.shape()[1..].iter().product::<usize>()
        } else {
            0
        };

        Ok(Self {
            node_features,
            edge_index,
            edge_features,
            num_nodes,
            num_edges,
            node_feat_dim,
            edge_feat_dim,
            message_dim: node_feat_dim, // Simplified: message dim = node feat dim
            aggr_type,
        })
    }

    /// Execute the message passing operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();

        // Create output buffer
        let output_size = self.num_nodes * self.node_feat_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_edges: u32,
            node_feat_dim: u32,
            edge_feat_dim: u32,
            message_dim: u32,
            aggr_type: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
            node_feat_dim: self.node_feat_dim as u32,
            edge_feat_dim: self.edge_feat_dim as u32,
            message_dim: self.message_dim as u32,
            aggr_type: self.aggr_type,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MessagePassing Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let placeholder = device.placeholder_buffer();
        let mut dispatch = ComputeDispatch::new(device, "MessagePassing")
            .shader(
                include_str!("../shaders/math/message_passing_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.node_features.buffer())
            .storage_read(1, self.edge_index.buffer());

        if let Some(ref ef) = self.edge_features {
            dispatch = dispatch.storage_read(2, ef.buffer());
        }

        dispatch
            .storage_read(3, placeholder)
            .storage_read(4, placeholder)
            .storage_rw(5, &output_buffer)
            .uniform(6, &params_buffer)
            .dispatch_1d(self.num_nodes as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.node_feat_dim],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_passing_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 4;
        let num_edges = 3;
        let node_feat_dim = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * node_feat_dim],
            vec![num_nodes, node_feat_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = Tensor::from_vec_on(
            vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            vec![num_edges, 2],
            device.clone(),
        )
        .await
        .unwrap();

        let output = MessagePassing::new(node_features, edge_index, None, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[num_nodes, node_feat_dim]);
    }
}
