// SPDX-License-Identifier: AGPL-3.0-or-later
//! Standard deviation reduction - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

mod compute;

#[cfg(test)]
mod tests;

use crate::error::Result;
use crate::tensor::Tensor;

/// Simple std reduction variant (scalar path).
pub const WGSL_STD_SIMPLE: &str = include_str!("../../shaders/misc/std_simple_f64.wgsl");

/// f64 canonical source for dimension-wise std.
pub(crate) const WGSL_STD_DIM_F64: &str = include_str!("../../shaders/reduce/std_dim_f64.wgsl");

/// f32 derived from f64 canonical source.
pub(crate) const WGSL_STD_DIM_F32: &str = WGSL_STD_DIM_F64;

/// Standard deviation reduction operation
pub struct Std {
    input: Tensor,
    dim: Option<usize>, // None = global std, Some(d) = std along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Std {
    /// Create a new std operation
    #[must_use]
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    /// Get the WGSL shader source for global reduction
    pub(crate) fn wgsl_shader_reduce() -> &'static str {
        include_str!("../../shaders/reduce/std_reduce.wgsl")
    }

    /// Get the WGSL shader source for dimension-wise reduction
    pub(crate) fn wgsl_shader_dim() -> &'static str {
        WGSL_STD_DIM_F32
    }

    /// Execute the std operation
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range (for dimension-wise std), or if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        compute::execute(self)
    }
}

impl Tensor {
    /// Compute standard deviation (global reduction)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn std(&self) -> Result<Self> {
        Std::new(self.clone(), None, false).execute()
    }

    /// Compute standard deviation along a dimension
    /// # Arguments
    /// * `dim` - Dimension to compute std along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range, or buffer allocation/GPU dispatch/buffer readback fails (e.g. device lost).
    pub fn std_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Std::new(self.clone(), Some(dim), keepdim).execute()
    }
}
