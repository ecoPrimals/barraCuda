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
use crate::device::compute_pipeline::{BindingKind, CachedPipeline};
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
    cached: CachedPipeline,
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
        let cached = CachedPipeline::build(
            &device,
            "Dada2EStep",
            &module,
            "e_step",
            &[
                BindingKind::Uniform,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRead,
                BindingKind::StorageRW,
            ],
        );
        Ok(Self { device, cached })
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
        let pbuf = self.device.create_uniform_buffer("Dada2Params", &params);
        let total_pairs = args.dimensions.n_seqs * args.dimensions.n_centers;
        self.cached.dispatch(
            &self.device,
            &[
                &pbuf,
                args.buffers.bases,
                args.buffers.quals,
                args.buffers.lengths,
                args.buffers.center_indices,
                args.buffers.log_err,
                args.buffers.scores,
            ],
            total_pairs.div_ceil(WORKGROUP_SIZE_1D),
        );
        Ok(())
    }
}
