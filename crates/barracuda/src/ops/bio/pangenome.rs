// SPDX-License-Identifier: AGPL-3.0-only

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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
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
        let bgl = super::snp::make_bgl(&device, &[true, false, false]);
        let layout = super::snp::make_layout(&device, &bgl, "PangenomeClassify");
        let pipeline =
            super::snp::make_pipeline(&device, &layout, &module, "main", "PangenomeClassify");
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
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
        let pbuf = super::snp::upload_uniform(&self.device, &params);
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    super::snp::bg_entry(0, &pbuf),
                    super::snp::bg_entry(1, presence),
                    super::snp::bg_entry(2, class_out),
                    super::snp::bg_entry(3, count_out),
                ],
            });
        super::snp::submit(
            &self.device,
            &self.pipeline,
            &bg,
            n_genes.div_ceil(WORKGROUP_SIZE_1D),
        );
        Ok(())
    }
}
