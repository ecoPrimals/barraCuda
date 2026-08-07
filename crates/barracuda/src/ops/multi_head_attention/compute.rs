// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Multi-Head Attention
//!
//! This module contains shader sources for:
//! 1. Projection pass: Project Q, K, V through weight matrices
//! 2. Output projection pass: Project concatenated heads through output matrix

use super::{MHAProjectionParams, MultiHeadAttention};
use crate::device::WgpuDevice;

pub(super) const PROJECTION_SHADER: &str =
    include_str!("../../shaders/attention/mha_projection_f64.wgsl");

impl MultiHeadAttention {
    /// Get WGSL shader for MHA output projection
    pub(super) fn wgsl_shader_output() -> &'static str {
        include_str!("../../shaders/tensor/mha_output_f64.wgsl")
    }
}

/// Build shared projection uniform params buffer.
pub(super) fn create_projection_params_buffer(
    device: &WgpuDevice,
    op: &MultiHeadAttention,
) -> wgpu::Buffer {
    let params = MHAProjectionParams {
        batch_size: op.batch_size() as u32,
        seq_len: op.seq_len() as u32,
        d_model: op.d_model() as u32,
        num_heads: op.num_heads() as u32,
        head_dim: op.head_dim() as u32,
    };

    device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MHA Projection Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}
