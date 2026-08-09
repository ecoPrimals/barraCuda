// SPDX-License-Identifier: AGPL-3.0-or-later

//! Position-parallel SNP calling on GPU.
//!
//! One thread per alignment column. Each thread counts allele frequencies
//! across all sequences, determines the reference allele (most common),
//! and flags polymorphic positions.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `snp_calling_f64.wgsl` — 5/5 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SnpParams {
    alignment_length: u32,
    n_sequences: u32,
    min_depth: u32,
    _pad: u32,
}

/// Position-parallel SNP calling on GPU.
pub struct SnpCallingF64 {
    device: Arc<WgpuDevice>,
}

impl SnpCallingF64 {
    /// Create SNP calling pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch SNP calling for alignment.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(
        &self,
        alignment_length: u32,
        n_sequences: u32,
        min_depth: u32,
        sequences: &wgpu::Buffer,
        is_variant: &wgpu::Buffer,
        ref_allele: &wgpu::Buffer,
        depth_out: &wgpu::Buffer,
        alt_freq_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = SnpParams {
            alignment_length,
            n_sequences,
            min_depth,
            _pad: 0,
        };
        let pbuf = self.device.create_uniform_buffer("SnpParams", &params);

        ComputeDispatch::new(&self.device, "SnpCalling")
            .shader(
                include_str!("../../shaders/bio/snp_calling_f64.wgsl"),
                "main",
            )
            .f64()
            .uniform(0, &pbuf)
            .storage_read(1, sequences)
            .storage_rw(2, is_variant)
            .storage_rw(3, ref_allele)
            .storage_rw(4, depth_out)
            .storage_rw(5, alt_freq_out)
            .dispatch_1d(alignment_length)
            .submit()?;

        Ok(())
    }
}
