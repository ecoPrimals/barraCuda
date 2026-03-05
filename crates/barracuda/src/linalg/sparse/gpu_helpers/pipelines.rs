// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pipeline creation and dispatch helpers for sparse GPU solvers.
//!
//! Single responsibility: compute pipeline construction for SpMV, dot, reduce,
//! and CG update steps. Shared by CG and BiCGSTAB solvers.

use crate::device::WgpuDevice;
use std::sync::Arc;

/// CG-specific pipelines (SpMV, dot, reduce, update_xr, update_p, alpha, beta).
pub struct CgPipelineSet {
    /// Sparse matrix-vector product pipeline
    pub spmv: wgpu::ComputePipeline,
    /// Dot product pipeline
    pub dot: wgpu::ComputePipeline,
    /// Final reduction pipeline
    pub reduce: wgpu::ComputePipeline,
    /// Update x and r vectors
    pub update_xr: wgpu::ComputePipeline,
    /// Update search direction p
    pub update_p: wgpu::ComputePipeline,
    /// Compute CG step size alpha
    pub compute_alpha: wgpu::ComputePipeline,
    /// Compute CG step size beta
    pub compute_beta: wgpu::ComputePipeline,
}

impl CgPipelineSet {
    /// Create all CG pipelines from spmv/dot shader and cg_kernels shader
    pub fn new(
        device: &Arc<WgpuDevice>,
        spmv_shader: &wgpu::ShaderModule,
        dot_reduce_shader: &wgpu::ShaderModule,
        cg_kernels_shader: &wgpu::ShaderModule,
        spmv_bgl: &wgpu::BindGroupLayout,
        dot_bgl: &wgpu::BindGroupLayout,
        reduce_bgl: &wgpu::BindGroupLayout,
        update_xr_bgl: &wgpu::BindGroupLayout,
        update_p_bgl: &wgpu::BindGroupLayout,
        compute_alpha_bgl: &wgpu::BindGroupLayout,
        compute_beta_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let spmv = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SpMV f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("SpMV PL"),
                        bind_group_layouts: &[spmv_bgl],
                        immediate_size: 0,
                    },
                )),
                module: spmv_shader,
                entry_point: Some("spmv_f64"),
                cache: None,
                compilation_options: Default::default(),
            });
        let dot = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dot f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Dot PL"),
                        bind_group_layouts: &[dot_bgl],
                        immediate_size: 0,
                    },
                )),
                module: dot_reduce_shader,
                entry_point: Some("dot_f64"),
                cache: None,
                compilation_options: Default::default(),
            });
        let reduce = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Final reduce f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Reduce PL"),
                        bind_group_layouts: &[reduce_bgl],
                        immediate_size: 0,
                    },
                )),
                module: dot_reduce_shader,
                entry_point: Some("final_reduce_f64"),
                cache: None,
                compilation_options: Default::default(),
            });
        let update_xr = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CG update xr"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Update xr PL"),
                        bind_group_layouts: &[update_xr_bgl],
                        immediate_size: 0,
                    },
                )),
                module: cg_kernels_shader,
                entry_point: Some("cg_update_xr"),
                cache: None,
                compilation_options: Default::default(),
            });
        let update_p = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CG update p"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Update p PL"),
                        bind_group_layouts: &[update_p_bgl],
                        immediate_size: 0,
                    },
                )),
                module: cg_kernels_shader,
                entry_point: Some("cg_update_p"),
                cache: None,
                compilation_options: Default::default(),
            });
        let compute_alpha =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute alpha"),
                    layout: Some(&device.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Compute alpha PL"),
                            bind_group_layouts: &[compute_alpha_bgl],
                            immediate_size: 0,
                        },
                    )),
                    module: cg_kernels_shader,
                    entry_point: Some("compute_alpha"),
                    cache: None,
                    compilation_options: Default::default(),
                });
        let compute_beta =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute beta"),
                    layout: Some(&device.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Compute beta PL"),
                            bind_group_layouts: &[compute_beta_bgl],
                            immediate_size: 0,
                        },
                    )),
                    module: cg_kernels_shader,
                    entry_point: Some("compute_beta"),
                    cache: None,
                    compilation_options: Default::default(),
                });
        Self {
            spmv,
            dot,
            reduce,
            update_xr,
            update_p,
            compute_alpha,
            compute_beta,
        }
    }
}

/// Helper to dispatch a compute pass (reduces verbosity in CG iteration).
#[inline(always)]
pub fn cg_dispatch_pass(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
    z: u32,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, Some(bind_group), &[]);
    pass.dispatch_workgroups(x, y, z);
}

/// Pipeline builder for sparse operations.
pub struct SparsePipelines {
    /// Sparse matrix-vector product pipeline
    pub spmv: wgpu::ComputePipeline,
    /// Dot product pipeline
    pub dot: wgpu::ComputePipeline,
    /// Final reduction pipeline
    pub reduce: wgpu::ComputePipeline,
}

impl SparsePipelines {
    /// Create common sparse pipelines from shader module.
    pub fn new(
        device: &Arc<WgpuDevice>,
        shader: &wgpu::ShaderModule,
        spmv_bgl: &wgpu::BindGroupLayout,
        dot_bgl: &wgpu::BindGroupLayout,
        reduce_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let spmv = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SpMV f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("SpMV PL"),
                        bind_group_layouts: &[spmv_bgl],
                        immediate_size: 0,
                    },
                )),
                module: shader,
                entry_point: Some("spmv_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        let dot = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dot f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Dot PL"),
                        bind_group_layouts: &[dot_bgl],
                        immediate_size: 0,
                    },
                )),
                module: shader,
                entry_point: Some("dot_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        let reduce = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Final reduce f64"),
                layout: Some(&device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Reduce PL"),
                        bind_group_layouts: &[reduce_bgl],
                        immediate_size: 0,
                    },
                )),
                module: shader,
                entry_point: Some("final_reduce_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        Self { spmv, dot, reduce }
    }
}
