// SPDX-License-Identifier: AGPL-3.0-only
//! Sparse linear algebra for large-scale scientific computing
//!
//! Provides sparse matrix representations and solvers for problems where
//! dense matrices would be prohibitively expensive. Essential for:
//! - Large HFB basis sets (nuclear physics)
//! - Finite element methods
//! - Graph algorithms
//! - Machine learning on sparse data
//!
//! # Precision Philosophy
//!
//! **Both CPU and GPU use f64 by default.**
//! GPU solvers use native WGSL f64 via SPIR-V/Vulkan, bypassing CUDA's
//! artificial fp64 throttle for 1:2-3 FP64:FP32 performance.
//!
//! # Storage Formats
//!
//! - **COO** (Coordinate): Easy construction, inefficient operations
//! - **CSR** (Compressed Sparse Row): Efficient row access, `SpMV`
//! - **CSC** (Compressed Sparse Column): Efficient column access
//!
//! # Solvers
//!
//! ## CPU Solvers
//! - **CG** (`cg_solve`): Conjugate Gradient for SPD matrices
//! - **`BiCGSTAB`** (`bicgstab_solve`): For general non-symmetric matrices
//! - **Jacobi** (`jacobi_solve`): Simple iterative method
//!
//! ## GPU Solvers (f64)
//! - **`CgGpu`** (`CgGpu::solve`): GPU-accelerated Conjugate Gradient (SPD)
//! - **`BiCgStabGpu`** (`BiCgStabGpu::solve`): GPU-accelerated `BiCGSTAB` (non-symmetric)
//!
//! ## Eigensolvers
//! - **`sparse_eigh`**: Lanczos-based eigenvalues for sparse symmetric matrices
//! - **`sparse_eigh_smallest`**: k smallest eigenvalues (extremal convergence)
//! - **`sparse_eigh_largest`**: k largest eigenvalues
//!
//! # Example
//!
//! ```
//! use barracuda::linalg::sparse::{CsrMatrix, cg_solve};
//!
//! // Build a sparse SPD matrix
//! let matrix = CsrMatrix::from_triplets(3, 3, &[
//!     (0, 0, 4.0), (0, 1, -1.0),
//!     (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
//!     (2, 1, -1.0), (2, 2, 4.0),
//! ]);
//!
//! let b = vec![1.0, 2.0, 3.0];
//! let x = cg_solve(&matrix, &b, 1e-10, 100)?;
//! # Ok::<(), barracuda::error::BarracudaError>(())
//! ```
//!
//! # Reference
//!
//! - hotSpring Phase 5 Handoff: Large HFB basis sets requirement
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems

pub mod bicgstab_gpu;
pub mod cg_gpu;
pub mod csr;
pub mod eigh;
pub mod gpu_helpers; // EVOLVED: Extracted common GPU helpers (Feb 14, 2026)
pub mod solvers;

pub use bicgstab_gpu::{BiCgStabGpu, BiCgStabGpuResult};
pub use cg_gpu::{CgGpu, CgGpuResult};
pub use csr::{CooMatrix, CsrMatrix};
pub use eigh::{SparseEighResult, sparse_eigh, sparse_eigh_largest, sparse_eigh_smallest};
pub use gpu_helpers::{
    CgPipelineSet, SparseBindGroupLayouts, SparseBuffers, SparsePipelines, cg_dispatch_pass,
};
pub use solvers::{SolverConfig, SolverResult, bicgstab_solve, cg_solve, jacobi_solve};
