// SPDX-License-Identifier: AGPL-3.0-only
//! GPU helpers for sparse linear algebra
//!
//! This module provides common utilities for GPU-accelerated sparse solvers,
//! extracted from `cg_gpu.rs` to enable code reuse across solvers.
//!
//! **Refactored (domain separation)**:
//! - `buffers`: Buffer creation, readback, copy (`SparseBuffers`)
//! - `bind_group_layouts`: BGL creation for `SpMV`, dot, reduce, CG steps (`SparseBindGroupLayouts`)
//! - `pipelines`: Pipeline creation and dispatch (`CgPipelineSet`, `SparsePipelines`, `cg_dispatch_pass`)

mod bind_group_layouts;
mod buffers;
mod pipelines;

pub use bind_group_layouts::SparseBindGroupLayouts;
pub use buffers::SparseBuffers;
pub use pipelines::{CgPipelineSet, SparsePipelines, cg_dispatch_pass};

#[cfg(test)]
mod tests {
    #[test]
    fn test_sparse_buffers_creation() {
        // Would need a device for actual testing
        // This test verifies the module compiles correctly
    }
}
