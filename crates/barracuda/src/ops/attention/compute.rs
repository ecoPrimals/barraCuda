// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Scaled Dot-Product Attention
//!
//! This module contains the 3-pass GPU execution:
//! 1. Pass 1: Compute QK^T scores (matrix multiplication)
//! 2. Pass 2: Apply softmax to scores (row-wise)
//! 3. Pass 3: Apply weights to values (weighted sum)

use super::{Attention, AttentionParams};
use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::Result;
use crate::tensor::Tensor;

impl Attention {
    /// Execute attention operation (3 GPU passes)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query().device();

        // Extract dimensions (cross-attention: Q and K/V may have different seq_len)
        let q_shape = self.query().shape();
        let kv_shape = self.key().shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let kv_seq_len = kv_shape[2];
        let head_dim = q_shape[3];

        let params = AttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            q_seq_len: q_seq_len as u32,
            kv_seq_len: kv_seq_len as u32,
            head_dim: head_dim as u32,
            _padding: [0; 3],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Params"),
            size: std::mem::size_of::<AttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Score matrix: [B, H, q_seq, kv_seq]
        let scores_size = batch_size * num_heads * q_seq_len * kv_seq_len;
        let scores_buffer = device.create_buffer_f32(scores_size)?;
        let weights_buffer = device.create_buffer_f32(scores_size)?;

        // Output: [B, H, q_seq, head_dim]
        let output_size = batch_size * num_heads * q_seq_len * head_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        const TILE_SIZE: u32 = 16;
        let workgroups_matmul_x = (kv_seq_len as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_matmul_y = (q_seq_len as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_matmul_z = (batch_size * num_heads) as u32;
        let workgroups_apply_x = (head_dim as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_apply_y = (q_seq_len as u32).div_ceil(TILE_SIZE).max(1);
        let workgroups_apply_z = (batch_size * num_heads) as u32;

        let mut batch = BatchedComputeDispatch::new(device);

        batch.push(
            ComputeDispatch::new(device, "Attention Matmul")
                .shader(Self::shader_matmul(), "main")
                .storage_read(0, self.query().buffer())
                .storage_read(1, self.key().buffer())
                .storage_rw(2, &scores_buffer)
                .uniform(3, &params_buffer)
                .dispatch(
                    workgroups_matmul_x,
                    workgroups_matmul_y,
                    workgroups_matmul_z,
                ),
        )?;

        batch.push(
            ComputeDispatch::new(device, "Attention Softmax")
                .shader(Self::shader_softmax(), "main")
                .storage_read(0, &scores_buffer)
                .storage_rw(1, &weights_buffer)
                .uniform(2, &params_buffer)
                .dispatch_1d((batch_size * num_heads * q_seq_len) as u32),
        )?;

        batch.push(
            ComputeDispatch::new(device, "Attention Apply")
                .shader(Self::shader_apply(), "main")
                .storage_read(0, &weights_buffer)
                .storage_read(1, self.value().buffer())
                .storage_rw(2, &output_buffer)
                .uniform(3, &params_buffer)
                .dispatch(workgroups_apply_x, workgroups_apply_y, workgroups_apply_z),
        )?;

        batch.submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, q_seq_len, head_dim],
            device.clone(),
        ))
    }
}
