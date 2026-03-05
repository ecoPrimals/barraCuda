// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triangular Solve - Forward/Backward Substitution - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Runtime-configured matrix size
//! - ✅ Capability-based dispatch
//!
//! ## Algorithm
//!
//! Solves triangular linear systems:
//! ```text
//! Forward:  L·x = b (lower triangular L)
//! Backward: Uᵀ·x = b (upper triangular U or transpose of lower)
//!
//! Used after Cholesky: A = L·Lᵀ
//! 1. Solve L·z = b (forward)
//! 2. Solve Lᵀ·x = z (backward)
//! Result: x solves A·x = b
//! ```
//!
//! ## Precision Support
//!
//! - `execute()` - f32 precision
//! - `TriangularSolveF64::execute()` - f64 precision (science-grade)
//! - `TriangularSolveF64::execute_transpose()` - Solve using Lᵀ (for Cholesky step 2)
//!
//! ## Use Case
//!
//! **RBF Surrogate Learning** (hotSpring physics integration):
//! - After Cholesky: K = L·Lᵀ
//! - Solve K·w = y → solve L·(Lᵀ·w) = y
//! - Step 1: L·z = y (forward) → z
//! - Step 2: Lᵀ·w = z (backward) → w
//! - Result: w are the RBF weights
//!
//! ## References
//!
//! - Golub & Van Loan, "Matrix Computations", Section 3.1
//! - Used in scipy.linalg.solve_triangular
//! - Completes the Cholesky solve pipeline

mod f32;
mod f64;
#[cfg(test)]
mod tests;

pub use f32::TriangularSolve;
pub use f64::TriangularSolveF64;
