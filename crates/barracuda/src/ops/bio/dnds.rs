// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch pairwise dN/dS (Nei-Gojobori 1986) on GPU.
//!
//! One thread per coding sequence pair. Classifies synonymous/nonsynonymous
//! sites and differences, then applies Jukes-Cantor correction.
//! Polyfill required for Ada Lovelace (uses f64 log in Jukes-Cantor).
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `dnds_batch_f64.wgsl` — 9/9 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/dnds_batch_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DnDsParams {
    n_pairs: u32,
    n_codons: u32,
}

/// Batch dN/dS computation on GPU.
pub struct DnDsBatchF64 {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl DnDsBatchF64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("dnds_batch_f64"));
        let bgl = super::snp::make_bgl(&device, &[true, true, true, false, false, false]);
        let layout = super::snp::make_layout(&device, &bgl, "DnDsBatch");
        let pipeline = super::snp::make_pipeline(&device, &layout, &module, "main", "DnDsBatch");
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    pub fn dispatch(
        &self,
        n_pairs: u32,
        n_codons: u32,
        seq_a: &wgpu::Buffer,
        seq_b: &wgpu::Buffer,
        genetic_code: &wgpu::Buffer,
        dn_out: &wgpu::Buffer,
        ds_out: &wgpu::Buffer,
        omega_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = DnDsParams { n_pairs, n_codons };
        let pbuf = super::snp::upload_uniform(&self.device, &params);
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    super::snp::bg_entry(0, &pbuf),
                    super::snp::bg_entry(1, seq_a),
                    super::snp::bg_entry(2, seq_b),
                    super::snp::bg_entry(3, genetic_code),
                    super::snp::bg_entry(4, dn_out),
                    super::snp::bg_entry(5, ds_out),
                    super::snp::bg_entry(6, omega_out),
                ],
            });
        super::snp::submit(&self.device, &self.pipeline, &bg, n_pairs.div_ceil(64));
        Ok(())
    }
}
