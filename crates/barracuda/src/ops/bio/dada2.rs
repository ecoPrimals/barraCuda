// SPDX-License-Identifier: AGPL-3.0-or-later

//! DADA2 E-step (batch `log_p_error`) on GPU.
//!
//! One thread per (sequence, center) pair. Sums precomputed
//! `log(err[from][to][qual])` over all alignment positions. No GPU
//! transcendentals — all log values precomputed on CPU.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `dada2_e_step.wgsl` — 88 pipeline checks PASS.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
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

/// Problem dimensions for a DADA2 E-step dispatch.
pub struct Dada2Dimensions {
    /// Number of input sequences.
    pub n_seqs: u32,
    /// Number of cluster centers.
    pub n_centers: u32,
    /// Maximum sequence length (for buffer sizing).
    pub max_len: u32,
}

/// GPU-resident buffers for a DADA2 E-step dispatch.
pub struct Dada2Buffers<'a> {
    /// Base calls `[n_seqs × max_len]`.
    pub bases: &'a wgpu::Buffer,
    /// Quality scores `[n_seqs × max_len]`.
    pub quals: &'a wgpu::Buffer,
    /// Per-sequence lengths `[n_seqs]`.
    pub lengths: &'a wgpu::Buffer,
    /// Center assignment indices `[n_seqs]`.
    pub center_indices: &'a wgpu::Buffer,
    /// Log-error model `[n_centers × max_len]`.
    pub log_err: &'a wgpu::Buffer,
    /// Output: log-probability scores `[n_seqs × n_centers]`.
    pub scores: &'a wgpu::Buffer,
}

/// Grouped arguments for [`Dada2EStepGpu::dispatch`].
pub struct Dada2DispatchArgs<'a> {
    /// Problem dimensions.
    pub dimensions: Dada2Dimensions,
    /// GPU buffers.
    pub buffers: Dada2Buffers<'a>,
}

/// DADA2 E-step: batch log-probability matrix on GPU.
pub struct Dada2EStepGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl Dada2EStepGpu {
    /// Creates a new DADA2 E-step GPU kernel for the given device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
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
    /// * `args.buffers.bases` — `[n_seqs × max_len]` u32 encoded bases
    /// * `args.buffers.quals` — `[n_seqs × max_len]` u32 phred scores
    /// * `args.buffers.lengths` — `[n_seqs]` u32 actual lengths
    /// * `args.buffers.center_indices` — `[n_centers]` u32 center sequence indices
    /// * `args.buffers.log_err` — `[4 × 4 × 42 = 672]` f64 precomputed log error table
    /// * `args.buffers.scores` — `[n_seqs × n_centers]` f64 output
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(&self, args: &Dada2DispatchArgs<'_>) -> Result<()> {
        let params = Dada2Params {
            n_seqs: args.dimensions.n_seqs,
            n_centers: args.dimensions.n_centers,
            max_len: args.dimensions.max_len,
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
                    super::snp::bg_entry(1, args.buffers.bases),
                    super::snp::bg_entry(2, args.buffers.quals),
                    super::snp::bg_entry(3, args.buffers.lengths),
                    super::snp::bg_entry(4, args.buffers.center_indices),
                    super::snp::bg_entry(5, args.buffers.log_err),
                    super::snp::bg_entry(6, args.buffers.scores),
                ],
            });
        let total_pairs = args.dimensions.n_seqs * args.dimensions.n_centers;
        super::snp::submit(
            &self.device,
            &self.pipeline,
            &bg,
            total_pairs.div_ceil(WORKGROUP_SIZE_1D),
        );
        Ok(())
    }
}
