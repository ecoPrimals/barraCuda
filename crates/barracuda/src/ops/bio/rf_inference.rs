// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch Random Forest GPU Inference
//!
//! One thread per (sample, tree) pair. Each thread traverses one decision tree
//! for one sample. Results stored in `[n_samples × n_trees]`, then reduced on
//! CPU for majority vote or averaging.
//!
//! `SoA` layout avoids bitcast — thresholds stored as native f64.
//!
//! Provenance: wetSpring handoff v5 → `ToadStool` absorption.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RfParams {
    n_samples: u32,
    n_trees: u32,
    n_nodes_max: u32,
    n_features: u32,
}

/// GPU-accelerated batch Random Forest inference.
///
/// Evaluates all trees across all samples in parallel on GPU.
/// CPU performs majority-vote reduction over tree predictions.
pub struct RfBatchInferenceGpu {
    device: Arc<WgpuDevice>,
}

impl RfBatchInferenceGpu {
    /// Creates a new batch Random Forest inference GPU kernel for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Run inference on GPU. Returns per-tree predictions `[n_samples × n_trees]`.
    ///
    /// # Arguments
    /// * `node_features_buf` — `[n_trees × n_nodes_max]` i32 (feature index, <0 = leaf)
    /// * `node_thresh_buf`   — `[n_trees × n_nodes_max]` f64 (split thresholds)
    /// * `node_children_buf` — `[n_trees × n_nodes_max × 2]` i32 (left/right or leaf class)
    /// * `features_buf`      — `[n_samples × n_features]` f64 (input features)
    /// * `predictions_buf`   — `[n_samples × n_trees]` u32 (output, written by kernel)
    /// # Errors
    ///
    /// Returns [`Err`] if shader compilation or GPU dispatch fails.
    pub fn dispatch(
        &self,
        node_features_buf: &wgpu::Buffer,
        node_thresh_buf: &wgpu::Buffer,
        node_children_buf: &wgpu::Buffer,
        features_buf: &wgpu::Buffer,
        predictions_buf: &wgpu::Buffer,
        n_samples: u32,
        n_trees: u32,
        n_nodes_max: u32,
        n_features: u32,
    ) -> Result<()> {
        let params = RfParams {
            n_samples,
            n_trees,
            n_nodes_max,
            n_features,
        };

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RfBatch params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let total = n_samples * n_trees;

        ComputeDispatch::new(&self.device, "RfBatch")
            .shader(
                include_str!("../../shaders/ml/rf_batch_inference.wgsl"),
                "main",
            )
            .uniform(0, &params_buf)
            .storage_read(1, node_features_buf)
            .storage_read(2, node_thresh_buf)
            .storage_read(3, node_children_buf)
            .storage_read(4, features_buf)
            .storage_rw(5, predictions_buf)
            .dispatch_1d(total)
            .submit()?;
        Ok(())
    }
}
