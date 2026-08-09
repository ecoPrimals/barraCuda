// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pangenome gene classification on GPU.
//!
//! One thread per gene cluster. Reads presence/absence across genomes
//! and classifies: core (all), accessory (2+), unique (1), absent (0).
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `pangenome_classify.wgsl` — 6/6 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::{BindingKind, CachedPipeline};
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/pangenome_classify.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PangenomeParams {
    n_genes: u32,
    n_genomes: u32,
}

/// Pangenome gene family classification on GPU.
pub struct PangenomeClassifyGpu {
    device: Arc<WgpuDevice>,
    cached: CachedPipeline,
}

impl PangenomeClassifyGpu {
    /// Create pangenome classifier.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("pangenome_classify"));
        let cached = CachedPipeline::build(
            &device,
            "PangenomeClassify",
            &module,
            "main",
            &[
                BindingKind::Uniform,
                BindingKind::StorageRead,
                BindingKind::StorageRW,
                BindingKind::StorageRW,
            ],
        );
        Ok(Self { device, cached })
    }

    /// Dispatch pangenome classification.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(
        &self,
        n_genes: u32,
        n_genomes: u32,
        presence: &wgpu::Buffer,
        class_out: &wgpu::Buffer,
        count_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = PangenomeParams { n_genes, n_genomes };
        let pbuf = self
            .device
            .create_uniform_buffer("PangenomeParams", &params);
        self.cached.dispatch(
            &self.device,
            &[&pbuf, presence, class_out, count_out],
            n_genes.div_ceil(WORKGROUP_SIZE_1D),
        );
        Ok(())
    }
}
