// SPDX-License-Identifier: AGPL-3.0-only

//! Batch Random Forest GPU Inference
//!
//! One thread per (sample, tree) pair. Each thread traverses one decision tree
//! for one sample. Results stored in `[n_samples × n_trees]`, then reduced on
//! CPU for majority vote or averaging.
//!
//! SoA layout avoids bitcast — thresholds stored as native f64.
//!
//! Provenance: wetSpring handoff v5 → ToadStool absorption.

use crate::device::WgpuDevice;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("../../shaders/ml/rf_batch_inference.wgsl");

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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl RfBatchInferenceGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let shader = device.compile_shader(SHADER, Some("RfBatchInference"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RfBatch BGL"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),  // node_features
                    storage_entry(2, true),  // node_thresh
                    storage_entry(3, true),  // node_children
                    storage_entry(4, true),  // features
                    storage_entry(5, false), // predictions
                ],
            });

        let pl = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RfBatch PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RfBatch Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        Self {
            device,
            pipeline,
            bgl,
        }
    }

    /// Run inference on GPU. Returns per-tree predictions `[n_samples × n_trees]`.
    ///
    /// # Arguments
    /// * `node_features_buf` — `[n_trees × n_nodes_max]` i32 (feature index, <0 = leaf)
    /// * `node_thresh_buf`   — `[n_trees × n_nodes_max]` f64 (split thresholds)
    /// * `node_children_buf` — `[n_trees × n_nodes_max × 2]` i32 (left/right or leaf class)
    /// * `features_buf`      — `[n_samples × n_features]` f64 (input features)
    /// * `predictions_buf`   — `[n_samples × n_trees]` u32 (output, written by kernel)
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
    ) {
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

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RfBatch BG"),
                layout: &self.bgl,
                entries: &[
                    bg_entry(0, &params_buf),
                    bg_entry(1, node_features_buf),
                    bg_entry(2, node_thresh_buf),
                    bg_entry(3, node_children_buf),
                    bg_entry(4, features_buf),
                    bg_entry(5, predictions_buf),
                ],
            });

        let total = n_samples * n_trees;
        let workgroups = total.div_ceil(256);

        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RfBatch Encoder"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RfBatch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.device.submit_and_poll(Some(encoder.finish()));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
