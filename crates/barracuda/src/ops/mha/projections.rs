// SPDX-License-Identifier: AGPL-3.0-or-later
//! Decomposed projection operations for Multi-Head Attention.
//!
//! Instead of fusing matmul + head reshape into a single kernel (which causes
//! GPU watchdog timeouts at production sizes due to O(d_model) per-thread loops),
//! we compose two validated primitives:
//!
//! 1. `Tensor::matmul` — tiled, shared-memory GPU matmul (validated across codebase)
//! 2. `head_split.wgsl` / `head_concat.wgsl` — pure data-movement reshapes (validated)
//!
//! This resolves S-03b: MHA projection hangs at (B=4, S=128, H=8, d=512).

use std::sync::Arc;

use super::{MhaParams, MultiHeadAttention};
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HeadReshapeParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
}

impl MultiHeadAttention {
    /// Project input through weight matrix, then split into attention heads.
    ///
    /// `[B, S, D] × [D, D] → [B, S, D] → head_split → [B, H, S, D/H]`
    ///
    /// Derives seq_len from the input tensor (supports cross-attention where
    /// K/V have different seq_len from Q).
    pub(super) fn project_with_head_split(
        &self,
        input: &Tensor,
        weight: &Tensor,
        params: &MhaParams,
    ) -> Result<Tensor> {
        let device = input.device();
        let b = params.batch_size as usize;
        let s = input.shape()[1]; // actual seq_len from tensor, not params
        let d = params.d_model as usize;
        let h = params.num_heads as usize;
        let hd = params.head_dim as usize;

        let input_2d = input.clone().reshape(vec![b * s, d])?;
        let weight_2d = weight.clone().reshape(vec![d, d])?;
        let projected_2d = input_2d.matmul(&weight_2d)?;
        let projected = projected_2d.reshape(vec![b, s, d])?;

        Self::dispatch_head_split(device, &projected, b, s, h, hd)
    }

    /// Concatenate attention heads, then project through output weight.
    ///
    /// `[B, H, S, D/H] → head_concat → [B, S, D] × [D, D] → [B, S, D]`
    ///
    /// Decomposed from the fused `mha_output.wgsl` that caused GPU hangs.
    pub(super) fn concat_and_project(
        &self,
        attention_out: &Tensor,
        w_o: &Tensor,
        params: &MhaParams,
    ) -> Result<Tensor> {
        let device = attention_out.device();
        let b = params.batch_size as usize;
        let s = attention_out.shape()[2]; // q_seq_len from attention output
        let d = params.d_model as usize;
        let h = params.num_heads as usize;
        let hd = params.head_dim as usize;

        // Step 1: Head concat — [B, H, S, D] → [B, S, H*D]
        let concatenated = Self::dispatch_head_concat(device, attention_out, b, s, h, hd)?;

        // Step 2: Matmul — [B*S, D] × [D, D] → [B*S, D] → [B, S, D]
        let concat_2d = concatenated.reshape(vec![b * s, d])?;
        let w_o_2d = w_o.clone().reshape(vec![d, d])?;
        let output_2d = concat_2d.matmul(&w_o_2d)?;
        output_2d.reshape(vec![b, s, d])
    }

    fn dispatch_head_split(
        device: &Arc<WgpuDevice>,
        input: &Tensor,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let total = batch_size * num_heads * seq_len * head_dim;
        let output_buffer = device.create_buffer_f32(total)?;

        let params = HeadReshapeParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("head_split params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(
            &crate::session::pipelines::HEAD_SPLIT_F32,
            Some("head_split"),
        );

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("head_split BGL"),
                entries: &[
                    Self::storage_entry(0, true),
                    Self::storage_entry(1, false),
                    Self::uniform_entry(2),
                ],
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("head_split BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline = Self::make_pipeline(device, &shader, &bgl, "head_split");
        let workgroups = (total as u32).div_ceil(WORKGROUP_SIZE_1D);

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("head_split"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("head_split"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, seq_len, head_dim],
            device.clone(),
        ))
    }

    fn dispatch_head_concat(
        device: &Arc<WgpuDevice>,
        input: &Tensor,
        batch_size: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let total = batch_size * num_heads * seq_len * head_dim;
        let d_model = num_heads * head_dim;
        let output_buffer = device.create_buffer_f32(total)?;

        let params = HeadReshapeParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("head_concat params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(
            &crate::session::pipelines::HEAD_CONCAT_F32,
            Some("head_concat"),
        );

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("head_concat BGL"),
                entries: &[
                    Self::storage_entry(0, true),
                    Self::storage_entry(1, false),
                    Self::uniform_entry(2),
                ],
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("head_concat BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline = Self::make_pipeline(device, &shader, &bgl, "head_concat");
        let workgroups = (total as u32).div_ceil(WORKGROUP_SIZE_1D);

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("head_concat"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("head_concat"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, seq_len, d_model],
            device.clone(),
        ))
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

    fn make_pipeline(
        device: &WgpuDevice,
        shader: &wgpu::ShaderModule,
        bgl: &wgpu::BindGroupLayout,
        label: &str,
    ) -> wgpu::ComputePipeline {
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                immediate_size: 0,
            });
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            })
    }
}
