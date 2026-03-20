// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unique - GPU hash-based unique element detection
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Hash-based unique detection, not CPU sort
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates inputs
//! - Modern idiomatic Rust: Result<T, E>
//!
//! Finds unique elements in a tensor using a GPU hash table approach.
//! The operation performs:
//! 1. Hash-based marking of unique elements
//! 2. Prefix sum to determine output positions
//! 3. Compaction of unique values

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

mod compute;

#[cfg(test)]
mod tests;

/// Parameters for unique operation
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct UniqueParams {
    pub(super) input_size: u32,
    pub(super) num_buckets: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

/// Unique operation - finds unique elements in a tensor
pub struct Unique {
    input: Tensor,
}

impl Unique {
    /// Create a new unique operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor) -> Result<Self> {
        if input.is_empty() {
            return Err(BarracudaError::invalid_op(
                "unique",
                "Cannot find unique elements in empty tensor",
            ));
        }

        Ok(Self { input })
    }

    /// WGSL shader source for unique marking and compaction.
    /// Uses a dedicated f32 shader because the hash function is structurally
    /// different: f32 bitcasts to u32 (4 bytes), f64 bitcasts to `vec2<u32>`
    /// (8 bytes). Simple text downcast cannot bridge this.
    pub(super) fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/misc/unique_f32.wgsl")
    }

    /// WGSL shader source for prefix sum computation
    pub(super) fn prefix_sum_shader() -> &'static str {
        include_str!("../../shaders/misc/prefix_sum.wgsl")
    }

    /// Get input tensor (for use by compute module)
    pub(super) fn input(&self) -> &Tensor {
        &self.input
    }

    /// Execute the unique operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        compute::execute(self)
    }
}
