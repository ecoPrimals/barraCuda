// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Scaled Dot-Product Attention
//!
//! This module contains the 3-pass GPU execution:
//! 1. Pass 1: Compute QK^T scores (matrix multiplication)
//! 2. Pass 2: Apply softmax to scores (row-wise)
//! 3. Pass 3: Apply weights to values (weighted sum)

use super::{Attention, AttentionParams};
use crate::device::{DeviceCapabilities, WorkloadType};
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

        // ═══════════════════════════════════════════════════════════
        // PASS 1: Compute QK^T scores
        // ═══════════════════════════════════════════════════════════

        let shader_matmul = device.compile_shader(Self::shader_matmul(), Some("AttentionMatmul"));

        let bgl_matmul = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Attention Matmul BGL"),
                entries: &[
                    // Query
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Key
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Scores (output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bg_matmul = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attention Matmul BG"),
            layout: &bgl_matmul,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.query().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.key().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_matmul =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Attention Matmul PL"),
                    bind_group_layouts: &[&bgl_matmul],
                    immediate_size: 0,
                });

        let pipeline_matmul =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Attention Matmul Pipeline"),
                    layout: Some(&pipeline_layout_matmul),
                    module: &shader_matmul,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Attention Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attention Matmul Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_matmul);
            pass.set_bind_group(0, Some(&bg_matmul), &[]);

            let caps = DeviceCapabilities::from_device(device);
            let _optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            const TILE_SIZE: u32 = 16;
            // x = key positions, y = query positions
            let workgroups_x = (kv_seq_len as u32).div_ceil(TILE_SIZE).max(1);
            let workgroups_y = (q_seq_len as u32).div_ceil(TILE_SIZE).max(1);
            let workgroups_z = (batch_size * num_heads) as u32;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // ═══════════════════════════════════════════════════════════
        // PASS 2: Apply softmax
        // ═══════════════════════════════════════════════════════════

        let shader_softmax =
            device.compile_shader(Self::shader_softmax(), Some("AttentionSoftmax"));

        let bgl_softmax =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Attention Softmax BGL"),
                    entries: &[
                        // Scores (input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Weights (output)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bg_softmax = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attention Softmax BG"),
            layout: &bgl_softmax,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_softmax =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Attention Softmax PL"),
                    bind_group_layouts: &[&bgl_softmax],
                    immediate_size: 0,
                });

        let pipeline_softmax =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Attention Softmax Pipeline"),
                    layout: Some(&pipeline_layout_softmax),
                    module: &shader_softmax,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attention Softmax Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_softmax);
            pass.set_bind_group(0, Some(&bg_softmax), &[]);

            // Softmax: one thread per [batch, head, query_pos] row
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups =
                ((batch_size * num_heads * q_seq_len) as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        // ═══════════════════════════════════════════════════════════
        // PASS 3: Apply weights to values
        // ═══════════════════════════════════════════════════════════

        let shader_apply = device.compile_shader(Self::shader_apply(), Some("AttentionApply"));

        let bgl_apply = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Attention Apply BGL"),
                entries: &[
                    // Weights (input)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Value (input)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bg_apply = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attention Apply BG"),
            layout: &bgl_apply,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.value().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout_apply =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Attention Apply PL"),
                    bind_group_layouts: &[&bgl_apply],
                    immediate_size: 0,
                });

        let pipeline_apply =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Attention Apply Pipeline"),
                    layout: Some(&pipeline_layout_apply),
                    module: &shader_apply,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attention Apply Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline_apply);
            pass.set_bind_group(0, Some(&bg_apply), &[]);

            let caps = DeviceCapabilities::from_device(device);
            let _optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            const TILE_SIZE: u32 = 16;
            let workgroups_x = (head_dim as u32).div_ceil(TILE_SIZE).max(1);
            let workgroups_y = (q_seq_len as u32).div_ceil(TILE_SIZE).max(1);
            let workgroups_z = (batch_size * num_heads) as u32;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        // Submit all passes
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, num_heads, q_seq_len, head_dim],
            device.clone(),
        ))
    }
}
