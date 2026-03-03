// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Sparse Attention
//!
//! This module contains the 3-pass GPU execution:
//! 1. Pass 1: Compute QK^T scores (matmul) - reused from attention
//! 2. Pass 2: Apply softmax with sparse mask - new shader
//! 3. Pass 3: Apply weights to values - reused from attention

use super::{SparseAttention, SparseAttentionParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

impl SparseAttention {
    /// Execute sparse attention (3 GPU passes)
    ///
    /// **Deep Debt**: Reuses 2/3 shaders from validated attention!
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query().device();

        // Extract dimensions
        let shape = self.query().shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_len = shape[2];
        let head_dim = shape[3];

        // Create parameters
        let params = SparseAttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            stride: self.stride() as u32,
            _padding: [0, 0, 0],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Attention Params"),
            size: std::mem::size_of::<SparseAttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Intermediate buffers
        let scores_size = batch_size * num_heads * seq_len * seq_len;
        let scores_buffer = device.create_buffer_f32(scores_size)?;
        let weights_buffer = device.create_buffer_f32(scores_size)?;

        // Output buffer
        let output_size = batch_size * num_heads * seq_len * head_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Deep Debt Evolution: Capability-based dispatch
        let caps = DeviceCapabilities::from_device(device);

        // ═══════════════════════════════════════════════════════════
        // PASS 1: Compute QK^T scores (REUSED from attention ✅)
        // ═══════════════════════════════════════════════════════════
        let matmul_workgroups = ((batch_size * num_heads * seq_len * seq_len) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::MatMul));

        ComputeDispatch::new(device, "sparse_attn_matmul")
            .shader(Self::shader_matmul(), "main")
            .storage_read(0, self.query().buffer())
            .storage_read(1, self.key().buffer())
            .storage_rw(2, &scores_buffer)
            .uniform(3, &params_buffer)
            .dispatch(matmul_workgroups.max(1), 1, 1)
            .submit();

        // ═══════════════════════════════════════════════════════════
        // PASS 2: Apply softmax with sparse mask (NEW shader!)
        // ═══════════════════════════════════════════════════════════
        let softmax_workgroups = ((batch_size * num_heads * seq_len) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));

        ComputeDispatch::new(device, "sparse_attn_softmax")
            .shader(Self::shader_sparse_softmax(), "main")
            .storage_read(0, &scores_buffer)
            .storage_rw(1, &weights_buffer)
            .uniform(2, &params_buffer)
            .dispatch(softmax_workgroups.max(1), 1, 1)
            .submit();

        // ═══════════════════════════════════════════════════════════
        // PASS 3: Apply weights to values (REUSED from attention ✅)
        // ═══════════════════════════════════════════════════════════
        let apply_workgroups = ((batch_size * num_heads * seq_len * head_dim) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));

        ComputeDispatch::new(device, "sparse_attn_apply")
            .shader(Self::shader_apply(), "main")
            .storage_read(0, &weights_buffer)
            .storage_read(1, self.value().buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(apply_workgroups.max(1), 1, 1)
            .submit();

        // Return output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, seq_len, head_dim],
            device.clone(),
        ))
    }
}
