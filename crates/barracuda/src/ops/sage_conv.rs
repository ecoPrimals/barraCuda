// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SAGEConv` - `GraphSAGE` (Pure WGSL)
//!
//! Scalable sampling and aggregation: `h_i`' = W * [`h_i` || `aggr_i`]
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// `GraphSAGE` Convolution
pub struct SageConv {
    node_features: Tensor,
    edge_index: Vec<(usize, usize)>,
    weights: Tensor,
    degrees: Vec<u32>,
    num_nodes: usize,
    num_edges: usize,
    in_features: usize,
    out_features: usize,
    normalize: bool,
}

impl SageConv {
    /// Create a `GraphSAGE` convolution with node features, edge index, weights, and degrees.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `node_features` is not 2D, degrees length mismatch, weights shape invalid, or `edge_index` is empty.
    pub fn new(
        node_features: Tensor,
        edge_index: Vec<(usize, usize)>,
        weights: Tensor,
        degrees: Vec<u32>,
        normalize: bool,
    ) -> Result<Self> {
        let node_shape = node_features.shape();
        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "sage_conv",
                "node_features must be 2D [num_nodes, in_features]",
            ));
        }

        let num_nodes = node_shape[0];
        let in_features = node_shape[1];

        if degrees.len() != num_nodes {
            return Err(BarracudaError::invalid_op(
                "sage_conv",
                "degrees must have num_nodes elements",
            ));
        }

        let weight_shape = weights.shape();
        if weight_shape.len() != 2 || weight_shape[0] != 2 * in_features {
            return Err(BarracudaError::invalid_op(
                "sage_conv",
                "weights must be [2 * in_features, out_features]",
            ));
        }

        let out_features = weight_shape[1];

        let num_edges = edge_index.len();
        if num_edges == 0 {
            return Err(BarracudaError::invalid_op(
                "sage_conv",
                "edge_index cannot be empty",
            ));
        }

        Ok(Self {
            node_features,
            edge_index,
            weights,
            degrees,
            num_nodes,
            num_edges,
            in_features,
            out_features,
            normalize,
        })
    }

    /// Execute the `GraphSAGE` convolution and return the output node features.
    /// # Errors
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
                label: Some("SAGEConv Edge Index"),
                contents: bytemuck::cast_slice(&edge_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create degrees buffer
        let degrees_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SAGEConv Degrees"),
                contents: bytemuck::cast_slice(&self.degrees),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create aggregated features buffer (zero-initialized for atomic accumulation)
        let aggregated_size = self.num_nodes * self.in_features;
        let aggregated_buffer = device.create_buffer_f32(aggregated_size)?;

        // Create output buffer
        let output_size = self.num_nodes * self.out_features;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_nodes: u32,
            num_edges: u32,
            in_features: u32,
            out_features: u32,
            normalize: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            num_nodes: self.num_nodes as u32,
            num_edges: self.num_edges as u32,
            in_features: self.in_features as u32,
            out_features: self.out_features as u32,
            normalize: u32::from(self.normalize),
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SAGEConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shader_source = include_str!("../shaders/gnn/sage_conv_f64.wgsl");
        let mut batch = BatchedComputeDispatch::new(device);
        batch.push(
            ComputeDispatch::new(device, "SAGEConv Aggregate")
                .shader(shader_source, "aggregate")
                .uniform(0, &params_buffer)
                .storage_read(1, self.node_features.buffer())
                .storage_read(2, &edge_buffer)
                .storage_read(3, self.weights.buffer())
                .storage_read(4, &degrees_buffer)
                .storage_rw(5, &aggregated_buffer)
                .storage_rw(6, &output_buffer)
                .dispatch_1d(self.num_edges as u32),
        )?;
        batch.push(
            ComputeDispatch::new(device, "SAGEConv Transform")
                .shader(shader_source, "apply_transform")
                .uniform(0, &params_buffer)
                .storage_read(1, self.node_features.buffer())
                .storage_read(2, &edge_buffer)
                .storage_read(3, self.weights.buffer())
                .storage_read(4, &degrees_buffer)
                .storage_rw(5, &aggregated_buffer)
                .storage_rw(6, &output_buffer)
                .dispatch_1d(self.num_nodes as u32),
        )?;
        if self.normalize {
            batch.push(
                ComputeDispatch::new(device, "SAGEConv Normalize")
                    .shader(shader_source, "normalize_output")
                    .uniform(0, &params_buffer)
                    .storage_read(1, self.node_features.buffer())
                    .storage_read(2, &edge_buffer)
                    .storage_read(3, self.weights.buffer())
                    .storage_read(4, &degrees_buffer)
                    .storage_rw(5, &aggregated_buffer)
                    .storage_rw(6, &output_buffer)
                    .dispatch_1d(self.num_nodes as u32),
            )?;
        }
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
    async fn test_sage_conv_basic() {
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
        let degrees = vec![1u32; num_nodes];

        let weights = Tensor::from_vec_on(
            vec![0.1; 2 * in_features * out_features],
            vec![2 * in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let sage = SageConv::new(node_features, edge_index, weights, degrees, false).unwrap();
        let output = sage.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_sage_conv_with_normalize() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_nodes = 3;
        let in_features = 4;
        let out_features = 8;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * in_features],
            vec![num_nodes, in_features],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_index = vec![(0, 1), (1, 2)];
        let degrees = vec![1u32; num_nodes];

        let weights = Tensor::from_vec_on(
            vec![0.1; 2 * in_features * out_features],
            vec![2 * in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let sage = SageConv::new(node_features, edge_index, weights, degrees, true).unwrap();
        let output = sage.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }

    #[tokio::test]
    async fn test_sage_conv_large_batch() {
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

        let edge_index: Vec<(usize, usize)> =
            (0..num_nodes).map(|i| (i, (i + 1) % num_nodes)).collect();
        let degrees = vec![1u32; num_nodes];

        let weights = Tensor::from_vec_on(
            vec![0.1; 2 * in_features * out_features],
            vec![2 * in_features, out_features],
            device.clone(),
        )
        .await
        .unwrap();

        let sage = SageConv::new(node_features, edge_index, weights, degrees, false).unwrap();
        let output = sage.execute().unwrap();

        assert_eq!(output.shape(), &[num_nodes, out_features]);
    }
}
