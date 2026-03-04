// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Scaled Dot-Product Attention
//!
//! This module contains the GPU execution logic for scaled dot-product attention,
//! including the three-pass execution: matmul, softmax, and apply.

use super::{AttentionParams, ScaledDotProductAttention};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Execute the scaled dot-product attention operation
///
/// Performs multi-pass execution:
/// 1. Compute Q @ K^T scores
/// 2. Apply softmax to scores
/// 3. Apply attention weights to values
pub(super) fn execute_scaled_dot_product_attention(
    op: ScaledDotProductAttention,
) -> Result<Tensor> {
    let device = op.query().device();

    // Calculate buffer sizes
    let input_size = op.batch_size() * op.num_heads() * op.seq_len() * op.head_dim();
    let scores_size = op.batch_size() * op.num_heads() * op.seq_len() * op.seq_len();

    // Create intermediate buffers
    let scores_buffer = device.create_buffer_f32(scores_size)?;
    let weights_buffer = device.create_buffer_f32(scores_size)?;
    let output_buffer = device.create_buffer_f32(input_size)?;

    // Create parameters buffer (self-attention: q_seq_len == kv_seq_len)
    let params = AttentionParams {
        batch_size: op.batch_size() as u32,
        num_heads: op.num_heads() as u32,
        q_seq_len: op.seq_len() as u32,
        kv_seq_len: op.seq_len() as u32,
        head_dim: op.head_dim() as u32,
        _padding: [0; 3],
    };

    let params_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Attention Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    // Deep Debt Evolution: Capability-based dispatch
    let caps = DeviceCapabilities::from_device(device);
    const TILE_SIZE: u32 = 16;

    // ═══════════════════════════════════════════════════════════════
    // PASS 1: Compute Q @ K^T scores
    // ═══════════════════════════════════════════════════════════════
    let workgroups_x = (op.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
    let workgroups_y = (op.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
    let workgroups_z = (op.batch_size() * op.num_heads()) as u32;

    ComputeDispatch::new(device, "sdp_attn_matmul")
        .shader(ScaledDotProductAttention::wgsl_shader_matmul(), "main")
        .storage_read(0, op.query().buffer())
        .storage_read(1, op.key().buffer())
        .storage_rw(2, &scores_buffer)
        .uniform(3, &params_buffer)
        .dispatch(workgroups_x, workgroups_y, workgroups_z)
        .submit();

    // ═══════════════════════════════════════════════════════════════
    // PASS 2: Apply softmax to scores
    // ═══════════════════════════════════════════════════════════════
    let total_rows = op.batch_size() * op.num_heads() * op.seq_len();
    let softmax_workgroups =
        (total_rows as u32).div_ceil(caps.optimal_workgroup_size(WorkloadType::MatMul));

    ComputeDispatch::new(device, "sdp_attn_softmax")
        .shader(ScaledDotProductAttention::wgsl_shader_softmax(), "main")
        .storage_read(0, &scores_buffer)
        .storage_rw(1, &weights_buffer)
        .uniform(2, &params_buffer)
        .dispatch(softmax_workgroups, 1, 1)
        .submit();

    // ═══════════════════════════════════════════════════════════════
    // PASS 3: Apply attention weights to values
    // ═══════════════════════════════════════════════════════════════
    let apply_wg_x = (op.head_dim() as u32).div_ceil(TILE_SIZE).max(1);
    let apply_wg_y = (op.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
    let apply_wg_z = (op.batch_size() * op.num_heads()) as u32;

    ComputeDispatch::new(device, "sdp_attn_apply")
        .shader(ScaledDotProductAttention::wgsl_shader_apply(), "main")
        .storage_read(0, &weights_buffer)
        .storage_read(1, op.value().buffer())
        .storage_rw(2, &output_buffer)
        .uniform(3, &params_buffer)
        .dispatch(apply_wg_x, apply_wg_y, apply_wg_z)
        .submit();

    // Return output tensor
    Ok(Tensor::from_buffer(
        output_buffer,
        vec![op.batch_size(), op.num_heads(), op.seq_len(), op.head_dim()],
        device.clone(),
    ))
}
