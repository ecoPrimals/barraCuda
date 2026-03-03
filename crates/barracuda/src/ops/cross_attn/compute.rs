//! GPU compute operations for Cross Attention
//!
//! This module contains the 3-pass GPU execution:
//! 1. Pass 1: Compute QK^T scores (matmul)
//! 2. Pass 2: Apply softmax normalization
//! 3. Pass 3: Apply weights to values

use super::{CrossAttention, CrossAttentionParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

impl CrossAttention {
    /// Execute cross attention (3 GPU passes)
    ///
    /// **Deep Debt**: Custom WGSL handles asymmetric seq_lens
    pub fn execute(self) -> Result<Tensor> {
        let device = self.query().device();

        // Extract dimensions
        let q_shape = self.query().shape();
        let k_shape = self.key().shape();

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let decoder_seq = q_shape[2];
        let encoder_seq = k_shape[2]; // Different from decoder!
        let head_dim = q_shape[3];

        // Create parameters
        let params = CrossAttentionParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            decoder_seq: decoder_seq as u32,
            encoder_seq: encoder_seq as u32,
            head_dim: head_dim as u32,
            _padding: [0, 0, 0],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cross Attention Params"),
            size: std::mem::size_of::<CrossAttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Intermediate buffers
        let scores_size = batch_size * num_heads * decoder_seq * encoder_seq;
        let scores_buffer = device.create_buffer_f32(scores_size)?;
        let weights_buffer = device.create_buffer_f32(scores_size)?;

        // Output buffer [B, H, Dec, D]
        let output_size = batch_size * num_heads * decoder_seq * head_dim;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Deep Debt Evolution: Capability-based dispatch
        let caps = DeviceCapabilities::from_device(device);

        // ═══════════════════════════════════════════════════════════
        // PASS 1: Compute QK^T scores
        // ═══════════════════════════════════════════════════════════
        let matmul_workgroups = ((batch_size * num_heads * decoder_seq * encoder_seq) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::MatMul));

        ComputeDispatch::new(device, "cross_attn_matmul")
            .shader(Self::shader_matmul(), "main")
            .storage_read(0, self.query().buffer())
            .storage_read(1, self.key().buffer())
            .storage_rw(2, &scores_buffer)
            .uniform(3, &params_buffer)
            .dispatch(matmul_workgroups.max(1), 1, 1)
            .submit();

        // ═══════════════════════════════════════════════════════════
        // PASS 2: Apply softmax
        // ═══════════════════════════════════════════════════════════
        let softmax_workgroups = ((batch_size * num_heads * decoder_seq) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));

        ComputeDispatch::new(device, "cross_attn_softmax")
            .shader(Self::shader_softmax(), "main")
            .storage_read(0, &scores_buffer)
            .storage_rw(1, &weights_buffer)
            .uniform(2, &params_buffer)
            .dispatch(softmax_workgroups, 1, 1)
            .submit();

        // ═══════════════════════════════════════════════════════════
        // PASS 3: Apply weights to values
        // ═══════════════════════════════════════════════════════════
        let apply_workgroups = ((batch_size * num_heads * decoder_seq * head_dim) as u32)
            .div_ceil(caps.optimal_workgroup_size(WorkloadType::ElementWise));

        ComputeDispatch::new(device, "cross_attn_apply")
            .shader(Self::shader_apply(), "main")
            .storage_read(0, &weights_buffer)
            .storage_read(1, self.value().buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(apply_workgroups.max(1), 1, 1)
            .submit();

        // Return output tensor [B, H, Dec, D]
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, decoder_seq, head_dim],
            device.clone(),
        ))
    }
}
