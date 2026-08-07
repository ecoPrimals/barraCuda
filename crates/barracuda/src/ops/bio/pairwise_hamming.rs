// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pairwise Hamming Distance — GPU kernel.
//!
//! Computes the upper-triangle pairwise Hamming distance matrix for N
//! sequences of length L. Each thread handles one pair. Output is
//! N*(N-1)/2 normalized distances (proportion of differing sites).
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;

const WGSL_PAIRWISE_HAMMING: &str = include_str!("../../shaders/math/pairwise_hamming_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HammingParams {
    n_seqs: u32,
    seq_len: u32,
}

/// GPU pairwise Hamming distance computation.
pub struct PairwiseHammingGpu {
    device: Arc<WgpuDevice>,
}

impl PairwiseHammingGpu {
    /// Create pairwise Hamming distance calculator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute pairwise Hamming distances for `n_seqs` sequences of `seq_len`.
    ///
    /// `sequences_buf`: `[n_seqs × seq_len]` u32 (nucleotide codes)
    /// `distances_buf`: `[n_seqs*(n_seqs-1)/2]` f32 (normalized distances)
    #[expect(clippy::missing_panics_doc, reason = "dispatch submit is infallible on valid device")]
    pub fn dispatch(
        &self,
        sequences_buf: &wgpu::Buffer,
        distances_buf: &wgpu::Buffer,
        n_seqs: u32,
        seq_len: u32,
    ) {
        let d = self.device.device();

        let params = HammingParams { n_seqs, seq_len };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PairwiseHamming Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let n_pairs = n_seqs * (n_seqs - 1) / 2;

        ComputeDispatch::new(&self.device, "PairwiseHamming")
            .shader(WGSL_PAIRWISE_HAMMING, "pairwise_hamming")
            .storage_read(0, sequences_buf)
            .storage_rw(1, distances_buf)
            .uniform(2, &params_buf)
            .dispatch_1d(n_pairs)
            .submit()
            .expect("PairwiseHamming GPU dispatch failed");
    }
}
