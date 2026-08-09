// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pairwise Jaccard Distance — GPU kernel.
//!
//! Computes the upper-triangle Jaccard distance matrix for a pangenome
//! presence/absence (PA) matrix. Each thread handles one genome pair.
//!
//! Jaccard(i,j) = 1 - |intersection| / |union|
//!
//! PA matrix stored column-major: `pa[gene * n_genomes + genome]`.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

const WGSL_PAIRWISE_JACCARD: &str = include_str!("../../shaders/math/pairwise_jaccard_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct JaccardParams {
    n_genomes: u32,
    n_genes: u32,
}

/// GPU pairwise Jaccard similarity computation.
pub struct PairwiseJaccardGpu {
    device: Arc<WgpuDevice>,
}

impl PairwiseJaccardGpu {
    /// Create pairwise Jaccard similarity calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute pairwise Jaccard distances for a pangenome PA matrix.
    ///
    /// `pa_buf`: `[n_genes × n_genomes]` f32, column-major (1.0 = present, 0.0 = absent)
    /// `distances_buf`: `[n_genomes*(n_genomes-1)/2]` f32
    /// # Errors
    ///
    /// Returns [`Err`] if shader compilation or GPU dispatch fails.
    pub fn dispatch(
        &self,
        pa_buf: &wgpu::Buffer,
        distances_buf: &wgpu::Buffer,
        n_genomes: u32,
        n_genes: u32,
    ) -> Result<()> {
        let d = self.device.device();

        let params = JaccardParams { n_genomes, n_genes };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PairwiseJaccard Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let n_pairs = n_genomes * (n_genomes - 1) / 2;

        ComputeDispatch::new(&self.device, "PairwiseJaccard")
            .shader(WGSL_PAIRWISE_JACCARD, "pairwise_jaccard")
            .storage_read(0, pa_buf)
            .storage_rw(1, distances_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(n_pairs)
            .submit()?;
        Ok(())
    }
}
