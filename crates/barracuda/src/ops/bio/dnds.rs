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
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::{BindingKind, CachedPipeline};
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
    cached: CachedPipeline,
}

impl DnDsBatchF64 {
    /// Create dN/dS batch calculator.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("dnds_batch_f64"));
        let cached = CachedPipeline::build(
            &device,
            "DnDsBatch",
            &module,
            "main",
            &[
                BindingKind::Uniform,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRW,
                BindingKind::StorageRW,
                BindingKind::StorageRW,
            ],
        );
        Ok(Self { device, cached })
    }

    /// Dispatch dN/dS computation for codon sequence pairs.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
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
        let pbuf = self.device.create_uniform_buffer("DnDsParams", &params);
        self.cached.dispatch(
            &self.device,
            &[&pbuf, seq_a, seq_b, genetic_code, dn_out, ds_out, omega_out],
            n_pairs.div_ceil(WORKGROUP_SIZE_COMPACT),
        );
        Ok(())
    }
}
