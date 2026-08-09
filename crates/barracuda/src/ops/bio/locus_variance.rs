// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-Locus Allele Frequency Variance — GPU kernel.
//!
//! Computes population variance of allele frequencies across populations
//! for each locus independently. Core building block for Weir-Cockerham
//! FST estimation.
//!
//! Input:  `allele_freqs[pop * n_loci + locus]`
//! Output: `per_locus_var[locus]`
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// f64 canonical — f32 derived via `downcast_f64_to_f32` when needed.
pub const WGSL_LOCUS_VARIANCE_F64: &str = include_str!("../../shaders/bio/locus_variance_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
}

/// Per-locus allele frequency variance GPU kernel (f64 pipeline).
pub struct LocusVarianceGpu {
    device: Arc<WgpuDevice>,
}

impl LocusVarianceGpu {
    /// Creates a new per-locus allele frequency variance GPU kernel for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute per-locus allele frequency variance across populations.
    ///
    /// `allele_freqs_buf`: `[n_pops × n_loci]` f64
    /// `output_buf`:       `[n_loci]` f64
    /// # Errors
    ///
    /// Returns [`Err`] if shader compilation or GPU dispatch fails.
    pub fn dispatch(
        &self,
        allele_freqs_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        n_pops: u32,
        n_loci: u32,
    ) -> Result<()> {
        let d = self.device.device();

        let params = VarianceParams { n_pops, n_loci };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LocusVariance Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        ComputeDispatch::new(&self.device, "LocusVariance")
            .shader(WGSL_LOCUS_VARIANCE_F64, "locus_variance")
            .f64()
            .storage_read(0, allele_freqs_buf)
            .storage_rw(1, output_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(n_loci)
            .submit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_shader_contains_locus_variance() {
        assert!(WGSL_LOCUS_VARIANCE_F64.contains("fn locus_variance"));
        assert!(WGSL_LOCUS_VARIANCE_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _ = device.compile_shader_f64(WGSL_LOCUS_VARIANCE_F64, Some("locus_variance_f64"));
    }
}
