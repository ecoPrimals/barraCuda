// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GATConv` - Graph Attention Networks (Pure WGSL)
//!
//! Attention-based graph convolution with learnable attention coefficients
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Graph Attention Network Convolution
pub struct GatConv {
    node_features: Tensor,
    edge_index: Vec<(usize, usize)>,
    weight: Tensor,
    attention: Tensor,
    num_nodes: usize,
    num_edges: usize,
    in_features: usize,
    out_features: usize,
    leaky_slope: f32,
}

impl GatConv {
    /// Create GAT (Graph Attention) convolution.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        node_features: Tensor,
        edge_index: Vec<(usize, usize)>,
        weight: Tensor,
        attention: Tensor,
        leaky_slope: f32,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "gat_conv",
                "node_features must be 2D [num_nodes, in_features]",
            ));
        }

        let num_nodes = node_shape[0];
        let in_features = node_shape[1];

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 || weight_shape[0] != in_features {
            return Err(BarracudaError::invalid_op(
                "gat_conv",
                "weight must be [in_features, out_features]",
            ));
        }

        let out_features = weight_shape[1];

        let attention_shape = attention.shape();
        let attention_size = attention_shape.iter().product::<usize>();
        if attention_size != 2 * out_features {
            return Err(BarracudaError::invalid_op(
                "gat_conv",
                "attention must have 2 * out_features elements",
            ));
        }

        let num_edges = edge_index.len();
        if num_edges == 0 {
            return Err(BarracudaError::invalid_op(
                "gat_conv",
                "edge_index cannot be empty",
            ));
        }

        Ok(Self {
            node_features,
            edge_index,
            weight,
            attention,
            num_nodes,
            num_edges,
            in_features,
            out_features,
            leaky_slope,
        })
    }

    /// Execute GAT convolution.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        // Convert edge_index to u32 pairs
        let edge_data: Vec<u32> = self
            .edge_index
            .iter()
            .flat_map(|(src, dst)| vec![*src as u32, *dst as u32])
            .collect();

        // Create edge_index buffer
        let edge_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GATConv Edge Index"),
                contents: bytemuck::cast_slice(&edge_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create temporary transformed features buffer
        let transformed_size = self.num_nodes * self.out_features;
        let transformed_buffer = device.create_buffer_f32(transformed_size)?;

        // Create output buffer (zero-initialized for atomic accumulation)
        let output_buffer = device.create_buffer_f32(transformed_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_edges: u32,
            in_features: u32,
            out_features: u32,
            leaky_slope: f32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            leaky_slope: self.leaky_slope,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GATConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shader_source = include_str!("../shaders/gnn/gat_conv_f64.wgsl");
        let mut batch = BatchedComputeDispatch::new(device);
        batch.push(
            ComputeDispatch::new(device, "GATConv Transform")
                .shader(shader_source, "transform_features")
                .uniform(0, &params_buffer)
                .storage_read(1, self.node_features.buffer())
                .storage_read(2, &edge_buffer)
                .storage_read(3, self.weight.buffer())
                .storage_read(4, self.attention.buffer())
                .storage_rw(5, &transformed_buffer)
                .storage_rw(6, &output_buffer)
                .dispatch_1d(self.num_nodes as u32),
        )?;
        batch.push(
            ComputeDispatch::new(device, "GATConv Aggregate")
                .shader(shader_source, "aggregate")
                .uniform(0, &params_buffer)
                .storage_read(1, self.node_features.buffer())
                .storage_read(2, &edge_buffer)
                .storage_read(3, self.weight.buffer())
                .storage_read(4, self.attention.buffer())
                .storage_rw(5, &transformed_buffer)
                .storage_rw(6, &output_buffer)
                .dispatch_1d(self.num_edges as u32),
        )?;
        batch.submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.num_nodes, self.out_features],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gat_conv_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 4;
        let in_features = 8;
        let out_features = 16;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

        let weight = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let attention = Tensor::from_vec_on(
            vec![0.5; 2 * out_features],
            vec![2 * out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gat = GatConv::new(node_features, edge_index, weight, attention, 0.01).unwrap();
        let output = gat.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_gat_conv_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 2;
        let in_features = 4;
        let out_features = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = vec![(0, 1)];

        let weight = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let attention = Tensor::from_vec_on(
            vec![0.5; 2 * out_features],
            vec![2 * out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gat = GatConv::new(node_features, edge_index, weight, attention, 0.01).unwrap();
        let output = gat.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_gat_conv_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 100;
        let in_features = 64;
        let out_features = 128;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let mut edge_index = Vec::new();
        for i in 0..num_nodes {
            edge_index.push((i, (i + 1) % num_nodes));
        }

        let weight = Tensor::from_vec_on(
            vec![0.1; in_features * out_features],
            vec![in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let attention = Tensor::from_vec_on(
            vec![0.5; 2 * out_features],
            vec![2 * out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let gat = GatConv::new(node_features, edge_index, weight, attention, 0.01).unwrap();
        let output = gat.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }
}
