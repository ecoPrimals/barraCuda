// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transpose operation - N-Dimensional transpose with arbitrary permutations
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Supports both 2D transpose (swap last two dims) and N-D with permutation

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Transpose operation parameters (2D)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TransposeParams2D {
    pub rows: u32,
    pub cols: u32,
    pub _padding: [u32; 2],
}

/// Transpose operation
pub struct Transpose {
    input: Tensor,
    permutation: Option<Vec<usize>>,
}

impl Transpose {
    /// Create Transpose operation
    /// For 2D tensors: swaps rows and columns (default behavior)
    /// For N-D tensors: requires permutation vector specifying dimension order
    /// # Errors
    /// Returns [`Err`] if the input tensor is invalid.
    pub fn new(input: Tensor) -> Result<Self> {
        Ok(Self {
            input,
            permutation: None,
        })
    }

    /// Create Transpose operation with explicit permutation
    /// # Arguments
    /// * `input` - Input tensor
    /// * `permutation` - Dimension permutation (e.g., [0, 2, 1] swaps dims 1 and 2)
    /// # Errors
    /// Returns [`Err`] if permutation length does not match tensor rank, permutation
    /// contains invalid or duplicate indices, or the input tensor is invalid.
    pub fn with_permutation(input: Tensor, permutation: Vec<usize>) -> Result<Self> {
        let num_dims = input.shape().len();
        if permutation.len() != num_dims {
            return Err(BarracudaError::invalid_op(
                "Transpose",
                format!(
                    "Permutation length {} doesn't match tensor rank {}",
                    permutation.len(),
                    num_dims
                ),
            ));
        }

        // Validate permutation
        let mut seen = vec![false; num_dims];
        for &idx in &permutation {
            if idx >= num_dims {
                return Err(BarracudaError::invalid_op(
                    "Transpose",
                    format!("Invalid permutation index {idx} for rank {num_dims}"),
                ));
            }
            if seen[idx] {
                return Err(BarracudaError::invalid_op(
                    "Transpose",
                    format!("Duplicate index {idx} in permutation"),
                ));
            }
            seen[idx] = true;
        }

        Ok(Self {
            input,
            permutation: Some(permutation),
        })
    }

    /// WGSL shader source (embedded at compile time)
    pub(super) fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!(
                    "../../shaders/tensor/transpose_f64.wgsl"
                ).to_string()
            });
            &S
        }
    }

    /// Execute transpose on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn execute(self) -> Result<Tensor> {
        compute::execute_transpose(self.input, self.permutation)
    }
}

// Convenience method on Tensor
impl Tensor {
    /// Transpose tensor (swap last two dimensions for 2D, or use permutation for N-D)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn transpose(&self) -> Result<Self> {
        Transpose::new(self.clone())?.execute()
    }

    /// Transpose tensor with explicit permutation
    /// # Errors
    /// Returns [`Err`] if permutation is invalid or if buffer allocation, shader
    /// compilation, device, or compute submission fails.
    pub fn transpose_with_permutation(&self, permutation: Vec<usize>) -> Result<Self> {
        Transpose::with_permutation(self.clone(), permutation)?.execute()
    }
}
