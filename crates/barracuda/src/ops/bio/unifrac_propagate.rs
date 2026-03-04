// SPDX-License-Identifier: AGPL-3.0-or-later

//! UniFrac Tree Propagation — GPU kernel (f64).
//!
//! Bottom-up propagation of sample abundances through a CSR phylogenetic
//! tree. Two entry points:
//!   - `unifrac_leaf_init`: copy sample matrix into leaf node slots
//!   - `unifrac_propagate_level`: sum child contributions × branch length
//!
//! Multi-pass dispatch: leaf_init once, then propagate_level per tree level.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::WgpuDevice;

pub const WGSL_UNIFRAC_PROPAGATE: &str = include_str!("../../shaders/bio/unifrac_propagate.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniFracConfig {
    pub n_nodes: u32,
    pub n_samples: u32,
    pub n_leaves: u32,
    pub _pad: u32,
}

pub struct UniFracPropagateGpu {
    leaf_init_pipeline: wgpu::ComputePipeline,
    propagate_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl UniFracPropagateGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UniFrac BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UniFrac Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader_f64(WGSL_UNIFRAC_PROPAGATE, Some("UniFrac Shader"));

        let leaf_init_pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("UniFrac LeafInit Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("unifrac_leaf_init"),
            compilation_options: Default::default(),
            cache: None,
        });

        let propagate_pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("UniFrac Propagate Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("unifrac_propagate_level"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            leaf_init_pipeline,
            propagate_pipeline,
            bgl,
            device,
        }
    }

    /// Initialize leaf nodes from sample matrix.
    pub fn dispatch_leaf_init(
        &self,
        config: &UniFracConfig,
        parent_buf: &wgpu::Buffer,
        branch_len_buf: &wgpu::Buffer,
        sample_mat_buf: &wgpu::Buffer,
        node_sums_buf: &wgpu::Buffer,
    ) {
        let bg = self.create_bind_group(
            config,
            parent_buf,
            branch_len_buf,
            sample_mat_buf,
            node_sums_buf,
        );
        self.dispatch_pipeline(
            &self.leaf_init_pipeline,
            &bg,
            config.n_leaves.div_ceil(WORKGROUP_SIZE_COMPACT),
        );
    }

    /// Propagate one tree level (call bottom-up per level).
    pub fn dispatch_propagate_level(
        &self,
        config: &UniFracConfig,
        parent_buf: &wgpu::Buffer,
        branch_len_buf: &wgpu::Buffer,
        sample_mat_buf: &wgpu::Buffer,
        node_sums_buf: &wgpu::Buffer,
    ) {
        let bg = self.create_bind_group(
            config,
            parent_buf,
            branch_len_buf,
            sample_mat_buf,
            node_sums_buf,
        );
        self.dispatch_pipeline(
            &self.propagate_pipeline,
            &bg,
            config.n_nodes.div_ceil(WORKGROUP_SIZE_COMPACT),
        );
    }

    fn create_bind_group(
        &self,
        config: &UniFracConfig,
        parent_buf: &wgpu::Buffer,
        branch_len_buf: &wgpu::Buffer,
        sample_mat_buf: &wgpu::Buffer,
        node_sums_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let d = self.device.device();
        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac Config"),
            contents: bytemuck::bytes_of(config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UniFrac BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: branch_len_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sample_mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: node_sums_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn dispatch_pipeline(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        workgroups_x: u32,
    ) {
        let q = self.device.queue();

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("UniFrac Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("UniFrac Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(bg), &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contains_entry_points() {
        assert!(WGSL_UNIFRAC_PROPAGATE.contains("fn unifrac_leaf_init"));
        assert!(WGSL_UNIFRAC_PROPAGATE.contains("fn unifrac_propagate_level"));
    }
}
