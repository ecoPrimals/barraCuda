// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pipelines and bind group layouts for Jacobi eigh (shared by slices and buffer APIs)

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::BglBuilder;
use std::sync::Arc;

/// Pipelines and bind group layouts for Jacobi eigh (shared by slices and buffer APIs)
pub(crate) struct EighPipelines {
    pub init_bgl: wgpu::BindGroupLayout,
    pub _init_pl: wgpu::PipelineLayout,
    pub init_v_pipeline: wgpu::ComputePipeline,
    pub extract_pipeline: wgpu::ComputePipeline,
    pub sweep_bgl: wgpu::BindGroupLayout,
    pub _sweep_pl: wgpu::PipelineLayout,
    pub compute_angles_pipeline: wgpu::ComputePipeline,
    pub rotate_a_pipeline: wgpu::ComputePipeline,
    pub update_blocks_pipeline: wgpu::ComputePipeline,
    pub rotate_v_pipeline: wgpu::ComputePipeline,
}

/// Create `EighPipelines` from device and shader module
pub(crate) fn create_eigh_pipelines(
    device: &Arc<WgpuDevice>,
    shader: &wgpu::ShaderModule,
) -> EighPipelines {
    let init_bgl = BglBuilder::new("Batched Init V BGL")
        .uniform(0)
        .storage_rw(1)
        .storage_rw(2)
        .storage_rw(3)
        .build(&device.device);

    let init_pl = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Batched Init V PL"),
            bind_group_layouts: &[&init_bgl],
            immediate_size: 0,
        });

    let init_v_pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched Init V"),
            layout: Some(&init_pl),
            module: shader,
            entry_point: Some("batched_init_V"),
            cache: None,
            compilation_options: Default::default(),
        });

    let extract_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Batched Extract Eigenvalues"),
                layout: Some(&init_pl),
                module: shader,
                entry_point: Some("batched_extract_eigenvalues"),
                cache: None,
                compilation_options: Default::default(),
            });

    let sweep_bgl = BglBuilder::new("Parallel Sweep BGL")
        .uniform(0)
        .storage_rw(1)
        .storage_rw(2)
        .storage_rw(3)
        .build(&device.device);

    let sweep_pl = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Parallel Sweep PL"),
            bind_group_layouts: &[&sweep_bgl],
            immediate_size: 0,
        });

    let compute_angles_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Parallel Compute Angles"),
                layout: Some(&sweep_pl),
                module: shader,
                entry_point: Some("parallel_compute_angles"),
                cache: None,
                compilation_options: Default::default(),
            });

    let rotate_a_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Parallel Rotate A"),
                layout: Some(&sweep_pl),
                module: shader,
                entry_point: Some("parallel_rotate_A"),
                cache: None,
                compilation_options: Default::default(),
            });

    let update_blocks_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Parallel Update Blocks"),
                layout: Some(&sweep_pl),
                module: shader,
                entry_point: Some("parallel_update_blocks"),
                cache: None,
                compilation_options: Default::default(),
            });

    let rotate_v_pipeline =
        device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Parallel Rotate V"),
                layout: Some(&sweep_pl),
                module: shader,
                entry_point: Some("parallel_rotate_V"),
                cache: None,
                compilation_options: Default::default(),
            });

    EighPipelines {
        init_bgl,
        _init_pl: init_pl,
        init_v_pipeline,
        extract_pipeline,
        sweep_bgl,
        _sweep_pl: sweep_pl,
        compute_angles_pipeline,
        rotate_a_pipeline,
        update_blocks_pipeline,
        rotate_v_pipeline,
    }
}
