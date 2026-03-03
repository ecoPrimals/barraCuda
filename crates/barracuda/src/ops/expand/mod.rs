//! Expand - Broadcast tensor to larger shape - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Broadcasting expands singleton dimensions to match target shape:
//! ```text
//! Input:  [1, 3] → Output: [4, 3]
//! Repeats the input across the first dimension
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::Result;
use crate::tensor::Tensor;

/// Expand operation for broadcasting tensors to larger shapes
pub struct Expand {
    input: Tensor,
    target_shape: Vec<usize>,
}

impl Expand {
    /// Create a new expand operation
    pub fn new(input: Tensor, target_shape: Vec<usize>) -> Self {
        Self {
            input,
            target_shape,
        }
    }

    /// Execute the expand operation
    pub fn execute(self) -> Result<Tensor> {
        compute::execute_expand(self.input, self.target_shape)
    }
}

impl Tensor {
    /// Expand/broadcast tensor to target shape
    pub fn expand_wgsl(self, target_shape: Vec<usize>) -> Result<Self> {
        Expand::new(self, target_shape).execute()
    }
}
