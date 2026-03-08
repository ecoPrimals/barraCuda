// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NonZero` - GPU prefix sum implementation
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Uses `prefix_sum.wgsl` for GPU parallel scan
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates inputs
//! - Modern idiomatic Rust: Result<T, E>, no `unwrap()`
//!
//! ## Algorithm
//!
//! Finds indices of all non-zero elements in a tensor using GPU-accelerated operations:
//! 1. Convert input to boolean mask (f32 → u32)
//! 2. Compute prefix sum to determine output size
//! 3. Compact indices using prefix sum offsets
//! 4. Convert u32 indices to f32 for Tensor compatibility
//!
//! **Key Properties**:
//! - Utility operation for sparse tensor operations
//! - Efficient GPU parallel implementation
//! - Returns flat indices of non-zero elements

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// `NonZero` operation parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NonZeroParams {
    pub input_size: u32,
    pub _padding: [u32; 3],
}

/// `NonZero` operation - finds indices of non-zero elements
///
/// **Deep Debt**: Utility operation for sparse tensor operations
pub struct NonZero {
    input: Tensor,
}

impl NonZero {
    /// Create new `NonZero` operation
    ///
    /// **Deep Debt**: Validates input is not empty
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if input is empty.
    pub fn new(input: Tensor) -> Result<Self> {
        if input.is_empty() {
            return Err(BarracudaError::invalid_op(
                "nonzero",
                "Cannot find nonzero elements in empty tensor",
            ));
        }

        Ok(Self { input })
    }

    /// WGSL shader source for nonzero operation
    pub(super) fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../../shaders/misc/nonzero_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// WGSL shader source for prefix sum
    pub(super) fn prefix_sum_shader() -> &'static str {
        include_str!("../../shaders/misc/prefix_sum.wgsl")
    }

    /// WGSL shader source for mask conversion
    pub(super) fn mask_convert_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../../shaders/misc/mask_convert_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// WGSL shader source for u32 to f32 conversion
    pub(super) fn u32_to_f32_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../../shaders/misc/u32_to_f32_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Get input tensor
    pub(super) fn input(&self) -> &Tensor {
        &self.input
    }
}
