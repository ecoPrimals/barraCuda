// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Grouped Query Attention
//!
//! This module contains the 3-pass GPU execution:
//! 1. Pass 1: Compute Q @ K^T scores (with grouped KV heads)
//! 2. Pass 2: Apply softmax to scores
//! 3. Pass 3: Apply attention weights to values

use super::{GQAParams, GroupedQueryAttention};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

impl GroupedQueryAttention {
    /// Execute the grouped query attention operation
    ///
    /// Performs multi-pass execution adapted for grouped queries:
    /// 1. Compute Q @ K^T scores (with grouped KV heads)
    /// 2. Apply softmax to scores
    /// 3. Apply attention weights to values
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query().device();

        // Calculate buffer sizes
        // Scores: [batch, num_q_heads, seq_len, seq_len]
        let scores_size = self.batch_size() * self.num_q_heads() * self.seq_len() * self.seq_len();
        let weights_size = scores_size;
        let output_size = self.batch_size() * self.num_q_heads() * self.seq_len() * self.head_dim();

        // Create intermediate buffers
        let scores_buffer = device.create_buffer_f32(scores_size)?;
        let weights_buffer = device.create_buffer_f32(weights_size)?;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create parameters buffer
        let params = GQAParams {
            batch_size: self.batch_size() as u32,
            num_q_heads: self.num_q_heads() as u32,
            num_kv_heads: self.num_kv_heads() as u32,
            seq_len: self.seq_len() as u32,
            head_dim: self.head_dim() as u32,
            heads_per_group: self.heads_per_group() as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GQA Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Deep Debt Evolution: Capability-based dispatch
        let caps = DeviceCapabilities::from_device(device);
        const TILE_SIZE: u32 = 16;

        // ═══════════════════════════════════════════════════════════════
        // PASS 1: Compute Q @ K^T scores (with grouped KV heads)
        // ═══════════════════════════════════════════════════════════════
        let workgroups_x = (self.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_y = (self.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_z = (self.batch_size() * self.num_q_heads()) as u32;

        ComputeDispatch::new(device, "gqa_matmul")
            .shader(Self::wgsl_shader_matmul(), "main")
            .storage_read(0, self.query().buffer())
            .storage_read(1, self.key().buffer())
            .storage_rw(2, &scores_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // ═══════════════════════════════════════════════════════════════
        // PASS 2: Apply softmax to scores
        // ═══════════════════════════════════════════════════════════════
        let total_rows = self.batch_size() * self.num_q_heads() * self.seq_len();
        let softmax_workgroups =
            (total_rows as u32).div_ceil(caps.optimal_workgroup_size(WorkloadType::MatMul));

        ComputeDispatch::new(device, "gqa_softmax")
            .shader(Self::wgsl_shader_softmax(), "main")
            .storage_read(0, &scores_buffer)
            .storage_rw(1, &weights_buffer)
            .uniform(2, &params_buffer)
            .dispatch(softmax_workgroups, 1, 1)
            .submit()?;

        // ═══════════════════════════════════════════════════════════════
        // PASS 3: Apply attention weights to values
        // ═══════════════════════════════════════════════════════════════
        let apply_wg_x = (self.head_dim() as u32).div_ceil(TILE_SIZE).max(1);
        let apply_wg_y = (self.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
        let apply_wg_z = (self.batch_size() * self.num_q_heads()) as u32;

        ComputeDispatch::new(device, "gqa_apply")
            .shader(Self::wgsl_shader_apply(), "main")
            .storage_read(0, &weights_buffer)
            .storage_read(1, self.value().buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(apply_wg_x, apply_wg_y, apply_wg_z)
            .submit()?;

        // Return output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![
                self.batch_size(),
                self.num_q_heads(),
                self.seq_len(),
                self.head_dim(),
            ],
            device.clone(),
        ))
    }
}
