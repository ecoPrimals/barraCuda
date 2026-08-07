// SPDX-License-Identifier: AGPL-3.0-or-later

//! `UniFrac` Tree Propagation — GPU kernel (f64).
//!
//! Bottom-up propagation of sample abundances through a CSR phylogenetic
//! tree. Two entry points:
//!   - `unifrac_leaf_init`: copy sample matrix into leaf node slots
//!   - `unifrac_propagate_level`: sum child contributions × branch length
//!
//! Multi-pass dispatch: `leaf_init` once, then `propagate_level` per tree level.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;

/// WGSL source for `UniFrac` tree propagation (leaf init + `propagate_level`).
pub const WGSL_UNIFRAC_PROPAGATE: &str = include_str!("../../shaders/bio/unifrac_propagate.wgsl");

/// Configuration for `UniFrac` tree propagation.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniFracConfig {
    /// Total number of nodes in the tree.
    pub n_nodes: u32,
    /// Number of samples (columns in sample matrix).
    pub n_samples: u32,
    /// Number of leaf nodes.
    pub n_leaves: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

/// GPU pipeline for `UniFrac` tree propagation (leaf init + level-wise propagate).
pub struct UniFracPropagateGpu {
    device: Arc<WgpuDevice>,
}

impl UniFracPropagateGpu {
    /// Create the `UniFrac` propagation pipeline for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
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
        self.dispatch(
            config,
            parent_buf,
            branch_len_buf,
            sample_mat_buf,
            node_sums_buf,
            "unifrac_leaf_init",
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
        self.dispatch(
            config,
            parent_buf,
            branch_len_buf,
            sample_mat_buf,
            node_sums_buf,
            "unifrac_propagate_level",
            config.n_nodes.div_ceil(WORKGROUP_SIZE_COMPACT),
        );
    }

    fn dispatch(
        &self,
        config: &UniFracConfig,
        parent_buf: &wgpu::Buffer,
        branch_len_buf: &wgpu::Buffer,
        sample_mat_buf: &wgpu::Buffer,
        node_sums_buf: &wgpu::Buffer,
        entry_point: &str,
        workgroups_x: u32,
    ) {
        let d = self.device.device();
        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UniFrac Config"),
            contents: bytemuck::bytes_of(config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        ComputeDispatch::new(&self.device, "UniFrac")
            .shader(WGSL_UNIFRAC_PROPAGATE, entry_point)
            .f64()
            .uniform(0, &config_buf)
            .storage_read(1, parent_buf)
            .storage_read(2, branch_len_buf)
            .storage_read(3, sample_mat_buf)
            .storage_rw(4, node_sums_buf)
            .dispatch(workgroups_x, 1, 1)
            .submit()
            .expect("UniFrac GPU dispatch failed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contains_entry_points() {
        assert!(WGSL_UNIFRAC_PROPAGATE.contains("fn unifrac_leaf_init"));
        assert!(WGSL_UNIFRAC_PROPAGATE.contains("fn unifrac_propagate_level"));
    }
}
