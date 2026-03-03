// SPDX-License-Identifier: AGPL-3.0-only

//! DADA2 E-step (batch log_p_error) on GPU.
//!
//! One thread per (sequence, center) pair. Sums precomputed
//! `log(err[from][to][qual])` over all alignment positions. No GPU
//! transcendentals — all log values precomputed on CPU.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `dada2_e_step.wgsl` — 88 pipeline checks PASS.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/bio/dada2_e_step.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Dada2Params {
    n_seqs: u32,
    n_centers: u32,
    max_len: u32,
    _pad: u32,
}

/// DADA2 E-step: batch log-probability matrix on GPU.
pub struct Dada2EStepGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl Dada2EStepGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("dada2_e_step"));
        let bgl = super::snp::make_bgl(&device, &[true, true, true, true, true, false]);
        let layout = super::snp::make_layout(&device, &bgl, "Dada2EStep");
        let pipeline = super::snp::make_pipeline(&device, &layout, &module, "e_step", "Dada2EStep");
        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    /// Dispatch E-step computation.
    ///
    /// * `bases` — `[n_seqs × max_len]` u32 encoded bases
    /// * `quals` — `[n_seqs × max_len]` u32 phred scores
    /// * `lengths` — `[n_seqs]` u32 actual lengths
    /// * `center_indices` — `[n_centers]` u32 center sequence indices
    /// * `log_err` — `[4 × 4 × 42 = 672]` f64 precomputed log error table
    /// * `scores` — `[n_seqs × n_centers]` f64 output
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        n_seqs: u32,
        n_centers: u32,
        max_len: u32,
        bases: &wgpu::Buffer,
        quals: &wgpu::Buffer,
        lengths: &wgpu::Buffer,
        center_indices: &wgpu::Buffer,
        log_err: &wgpu::Buffer,
        scores: &wgpu::Buffer,
    ) -> Result<()> {
        let params = Dada2Params {
            n_seqs,
            n_centers,
            max_len,
            _pad: 0,
        };
        let pbuf = super::snp::upload_uniform(&self.device, &params);
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    super::snp::bg_entry(0, &pbuf),
                    super::snp::bg_entry(1, bases),
                    super::snp::bg_entry(2, quals),
                    super::snp::bg_entry(3, lengths),
                    super::snp::bg_entry(4, center_indices),
                    super::snp::bg_entry(5, log_err),
                    super::snp::bg_entry(6, scores),
                ],
            });
        let total_pairs = n_seqs * n_centers;
        super::snp::submit(&self.device, &self.pipeline, &bg, total_pairs.div_ceil(256));
        Ok(())
    }
}
