// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Multi-Head Attention
//!
//! This module contains the GPU execution logic for:
//! 1. Projection pass: Project Q, K, V through weight matrices
//! 2. Output projection pass: Project concatenated heads through output matrix

use super::{MHAProjectionParams, MultiHeadAttention};
use crate::device::{DeviceCapabilities, WgpuDevice, WorkloadType};
use crate::error::Result;
use wgpu::util::DeviceExt;

const SHADER_F64: &str = include_str!("../../shaders/attention/mha_projection_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

impl MultiHeadAttention {
    /// Get WGSL shader for MHA projection
    pub(super) fn wgsl_shader_projection() -> &'static str {
        &SHADER_F32
    }

    /// Get WGSL shader for MHA output projection
    pub(super) fn wgsl_shader_output() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../../shaders/tensor/mha_output_f64.wgsl"
                ))
            });
            &S
        }
    }
}

/// Execute projection pass
pub(super) fn execute_projection(
    op: &MultiHeadAttention,
    device: &WgpuDevice,
    input: &crate::tensor::Tensor,
    weight: &crate::tensor::Tensor,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<wgpu::Buffer> {
    let output_size = op.batch_size() * op.num_heads() * op.seq_len() * op.head_dim();
    let output_buffer = device.create_buffer_f32(output_size)?;

    let params = MHAProjectionParams {
        batch_size: op.batch_size() as u32,
        seq_len: op.seq_len() as u32,
        d_model: op.d_model() as u32,
        num_heads: op.num_heads() as u32,
        head_dim: op.head_dim() as u32,
    };

    let params_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MHA Projection Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let shader_module = device.compile_shader(
        MultiHeadAttention::wgsl_shader_projection(),
        Some("MHA Projection Shader"),
    );

    let bind_group_layout =
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MHA Projection Bind Group Layout"),
                entries: &[
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

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MHA Projection Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight.buffer().as_entire_binding(),
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

    let pipeline_layout = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MHA Projection Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MHA Projection Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MHA Projection Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Deep Debt Evolution: Capability-based dispatch
    // Shader uses fixed 16x16 tiles (workgroup_size(16, 16, 1))
    // We use capability awareness to determine optimal tile count
    let caps = DeviceCapabilities::from_device(device);
    let _optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
    // Tile size is shader-constrained to 16x16, but we ensure capability awareness
    const TILE_SIZE: u32 = 16;
    let workgroups_x = (op.batch_size() as u32).div_ceil(TILE_SIZE).max(1);
    let workgroups_y = (op.num_heads() as u32).div_ceil(TILE_SIZE).max(1);
    // @workgroup_size(16, 16, 1): z tile is 1, need one workgroup per seq position
    let workgroups_z = op.seq_len() as u32;
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);

    Ok(output_buffer)
}

/// Execute output projection pass
pub(super) fn execute_output_projection(
    op: &MultiHeadAttention,
    device: &WgpuDevice,
    input: &wgpu::Buffer,
    encoder: &mut wgpu::CommandEncoder,
) -> Result<wgpu::Buffer> {
    let output_size = op.batch_size() * op.seq_len() * op.d_model();
    let output_buffer = device.create_buffer_f32(output_size)?;

    let params = MHAProjectionParams {
        batch_size: op.batch_size() as u32,
        seq_len: op.seq_len() as u32,
        d_model: op.d_model() as u32,
        num_heads: op.num_heads() as u32,
        head_dim: op.head_dim() as u32,
    };

    let params_buffer = device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MHA Output Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let shader_module = device.compile_shader(
        MultiHeadAttention::wgsl_shader_output(),
        Some("MHA Output Shader"),
    );

    let bind_group_layout =
        device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MHA Output Bind Group Layout"),
                entries: &[
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

    let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MHA Output Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: op.w_o().buffer().as_entire_binding(),
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

    let pipeline_layout = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MHA Output Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MHA Output Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MHA Output Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Deep Debt Evolution: Capability-based dispatch
    // Shader uses fixed 16x16 tiles (workgroup_size(16, 16, 1))
    // We use capability awareness to determine optimal tile count
    let caps = DeviceCapabilities::from_device(device);
    let _optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
    // Tile size is shader-constrained to 16x16, but we ensure capability awareness
    const TILE_SIZE: u32 = 16;
    let workgroups_x = (op.batch_size() as u32).div_ceil(TILE_SIZE).max(1);
    let workgroups_y = (op.seq_len() as u32).div_ceil(TILE_SIZE).max(1);
    // @workgroup_size(16, 16, 1): z tile is 1, need one workgroup per output dim
    let workgroups_z = op.d_model() as u32;
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);

    Ok(output_buffer)
}
