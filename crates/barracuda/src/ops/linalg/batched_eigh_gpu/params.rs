// SPDX-License-Identifier: AGPL-3.0-only
//! Parameter structs for batched eigh shaders

use bytemuck::{Pod, Zeroable};

/// Parameters for batched eigh shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct BatchedEighParams {
    pub n: u32,
    pub batch_size: u32,
    pub max_sweeps: u32,
    pub _pad: u32,
}

/// Parameters for parallel sweep operations
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct ParallelSweepParams {
    pub n: u32,
    pub batch_size: u32,
    pub current_p: u32,
    pub current_q: u32,
}

/// Parameters for single-dispatch eigensolve (n ≤ 32)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SingleDispatchParams {
    pub n: u32,
    pub batch_size: u32,
    pub max_sweeps: u32,
    pub tolerance: f32,
}
