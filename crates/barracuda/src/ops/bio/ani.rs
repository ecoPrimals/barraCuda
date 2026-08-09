// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch pairwise Average Nucleotide Identity (ANI) on GPU.
//!
//! One thread per sequence pair. Counts identical non-gap bases across
//! alignment positions, producing ANI ∈ [0, 1].
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `ani_batch_f64.wgsl` — 7/7 GPU checks PASS.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AniParams {
    n_pairs: u32,
    max_seq_len: u32,
}

/// Batch ANI computation on GPU.
pub struct AniBatchF64 {
    device: Arc<WgpuDevice>,
}

impl AniBatchF64 {
    /// Creates a new batch ANI GPU kernel for the given device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch ANI computation on GPU-resident buffers.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(
        &self,
        n_pairs: u32,
        max_seq_len: u32,
        seq_a: &wgpu::Buffer,
        seq_b: &wgpu::Buffer,
        ani_out: &wgpu::Buffer,
        aligned_out: &wgpu::Buffer,
        identical_out: &wgpu::Buffer,
    ) -> Result<()> {
        let params = AniParams {
            n_pairs,
            max_seq_len,
        };
        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AniBatch:params"),
            size: std::mem::size_of::<AniParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(&self.device, "AniBatch")
            .shader(include_str!("../../shaders/bio/ani_batch_f64.wgsl"), "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, seq_a)
            .storage_read(2, seq_b)
            .storage_rw(3, ani_out)
            .storage_rw(4, aligned_out)
            .storage_rw(5, identical_out)
            .dispatch_1d(n_pairs)
            .submit()?;

        Ok(())
    }
}
