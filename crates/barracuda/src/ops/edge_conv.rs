// SPDX-License-Identifier: AGPL-3.0-or-later
//! Edge Convolution for Graph Neural Networks
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Learns edge features by aggregating neighbor information using CSR-format edges.
//!
//! Reference: "Dynamic Graph CNN for Learning on Point Clouds" by Wang et al. (2019)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeConvParams {
    num_nodes: u32,
    feature_dim: u32,
    output_dim: u32,
    num_edges: u32,
}

/// Edge convolution for graph neural networks (Dynamic Graph CNN).
pub struct EdgeConv {
    node_features: Tensor,
    /// CSR row offsets: [`num_nodes` + 1] entries
    edge_offsets: Tensor,
    /// CSR column indices: [`num_edges`] neighbor node indices
    edge_targets: Tensor,
    mlp_weight: Tensor,
    mlp_bias: Tensor,
    num_edges: u32,
}

impl EdgeConv {
    /// Create `EdgeConv` operation with CSR-format edge storage
    /// # Arguments
    /// * `node_features` - Node features [`num_nodes`, `feature_dim`]
    /// * `edge_offsets` - CSR row offsets [`num_nodes` + 1] (stored as f32, cast to u32 in shader)
    /// * `edge_targets` - CSR column indices [`num_edges`] (stored as f32, cast to u32 in shader)
    /// * `mlp_weight` - MLP weight matrix [`output_dim`, 2 * `feature_dim`]
    /// * `mlp_bias` - MLP bias vector [`output_dim`]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        node_features: Tensor,
        edge_offsets: Tensor,
        edge_targets: Tensor,
        mlp_weight: Tensor,
        mlp_bias: Tensor,
    ) -> Result<Self> {
        let num_edges = edge_targets.len() as u32;

        Ok(Self {
            node_features,
            edge_offsets,
            edge_targets,
            mlp_weight,
            mlp_bias,
            num_edges,
        })
    }

    /// Execute `EdgeConv` on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.node_features.device();
        let node_shape = self.node_features.shape();

        if node_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "EdgeConv",
                format!(
                    "node_features must be 2D [num_nodes, feature_dim], got shape {node_shape:?}"
                ),
            ));
        }

        let num_nodes = node_shape[0];
        let feature_dim = node_shape[1];
        let output_dim = self.mlp_bias.len();

        // Create output buffer: [num_nodes, output_dim]
        let output_size = num_nodes * output_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = EdgeConvParams {
            num_nodes: num_nodes as u32,
            feature_dim: feature_dim as u32,
            output_dim: output_dim as u32,
            num_edges: self.num_edges,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("EdgeConv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "EdgeConv")
            .shader(
                include_str!("../shaders/gnn/edge_conv_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.node_features.buffer())
            .storage_read(1, self.edge_offsets.buffer())
            .storage_read(2, self.edge_targets.buffer())
            .storage_read(3, self.mlp_weight.buffer())
            .storage_read(4, self.mlp_bias.buffer())
            .storage_rw(5, &output_buffer)
            .uniform(6, &params_buffer)
            .dispatch_1d(num_nodes as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_nodes, output_dim],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_edge_conv_basic() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let num_nodes = 5;
        let feature_dim = 3;
        let output_dim = 4;

        let node_features = Tensor::from_vec_on(
            vec![1.0; num_nodes * feature_dim],
            vec![num_nodes, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        // CSR format: chain graph 0→1→2→3→4
        // edge_offsets: [0, 1, 2, 3, 4, 4] (node 4 has no outgoing edges)
        // edge_targets: [1, 2, 3, 4]
        let edge_offsets = Tensor::from_vec_on(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0], // num_nodes + 1 entries
            vec![num_nodes + 1],
            device.clone(),
        )
        .await
        .unwrap();

        let edge_targets = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0], // 4 edges
            vec![4],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_weight = Tensor::from_vec_on(
            vec![0.1; output_dim * 2 * feature_dim],
            vec![output_dim, 2 * feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let mlp_bias = Tensor::from_vec_on(vec![0.0; output_dim], vec![output_dim], device.clone())
            .await
            .unwrap();

        let result = EdgeConv::new(
            node_features,
            edge_offsets,
            edge_targets,
            mlp_weight,
            mlp_bias,
        )
        .unwrap()
        .execute()
        .unwrap();

        assert_eq!(result.shape(), &[num_nodes, output_dim]);
    }
}
