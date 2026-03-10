// SPDX-License-Identifier: AGPL-3.0-only
//! Linear algebra operations for scientific computing
//!
//! This module provides f64 linear algebra operations dispatched to GPU
//! shaders. All production math uses GPU (no CPU-only paths).
//!
//! # Available Operations
//!
//! ## Dense Decompositions
//!
//! - [`lu_decompose`] - LU decomposition with partial pivoting (PA = LU)
//! - [`qr_decompose`] - QR decomposition via Householder reflections
//! - [`svd_decompose`] - Singular value decomposition (A = `UΣVᵀ`)
//! - [`cholesky_f64`] - Cholesky decomposition for SPD matrices (A = `LLᵀ`)
//! - [`eigh_f64`] - Eigendecomposition for symmetric matrices (A = `VDVᵀ`)
//! - [`gen_eigh_f64`] - Generalized eigenvalue problem Ax = λBx
//!
//! ## Dense Solvers
//!
//! - [`solve_f64`] - General linear solve via Gauss-Jordan (GPU)
//! - [`solve_f64_cpu`] - CPU fallback for small matrices (no GPU required)
//! - [`lu_solve`] - Linear solve via LU decomposition
//! - [`qr_least_squares`] - Least squares via QR
//! - [`tridiagonal_solve_f64`] - Thomas algorithm for tridiagonal systems
//!
//! ## Sparse Operations ([`sparse`] module)
//!
//! For large-scale problems (HFB basis sets, finite elements):
//!
//! - [`sparse::CsrMatrix`] - Compressed sparse row format
//! - [`sparse::CooMatrix`] - Coordinate format (for construction)
//! - [`sparse::cg_solve`] - Conjugate gradient (SPD matrices)
//! - [`sparse::bicgstab_solve`] - `BiCGSTAB` (general matrices)
//! - [`sparse::jacobi_solve`] - Jacobi iteration
//!
//! ## Utilities
//!
//! - [`lu_det`] - Determinant via LU
//! - [`lu_inverse`] - Matrix inverse via LU
//! - [`svd_pinv`] - Pseudoinverse via SVD
//!
//! # Examples
//!
//! ```no_run
//! use barracuda::linalg::{solve_f64, lu_decompose, qr_decompose};
//! use barracuda::prelude::WgpuDevice;
//! use std::sync::Arc;
//!
//! # async fn example() -> barracuda::error::Result<()> {
//! let device = Arc::new(WgpuDevice::new().await?);
//! let a = vec![2.0, 1.0, 1.0, 3.0];
//! let b = vec![5.0, 8.0];
//! let x = solve_f64(device.clone(), &a, &b, 2)?;
//! // LU decomposition for multiple solves
//! let lu = lu_decompose(&a, 2)?;
//! let _det = lu.det();
//! let _x2 = lu.solve(&b)?;
//! // QR for least squares
//! let a_overdetermined = vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
//! let _qr = qr_decompose(&a_overdetermined, 3, 2)?;
//! # Ok(())
//! # }
//! ```

// CPU-only linear algebra (always available)
pub mod nmf;
pub mod ridge;

pub use nmf::{NmfConfig, NmfObjective, NmfResult};
pub use nmf::{cosine_similarity, nmf, relative_reconstruction_error, top_k_predictions};
pub use ridge::{RidgeResult, ridge_regression};

// GPU-dependent linear algebra (requires "gpu" feature)
#[cfg(feature = "gpu")]
pub mod cholesky;
#[cfg(feature = "gpu")]
pub mod eigh;
#[cfg(feature = "gpu")]
pub mod gen_eigh;
pub mod graph;
#[cfg(feature = "gpu")]
pub mod solve;
#[cfg(feature = "gpu")]
pub mod sparse;

#[cfg(feature = "gpu")]
pub use crate::ops::linalg::tridiagonal_solve;
#[cfg(feature = "gpu")]
pub use crate::ops::linalg::tridiagonal_solve as tridiagonal_solve_f64;
#[cfg(feature = "gpu")]
pub use crate::ops::linalg::{LuDecomposition, lu_decompose, lu_det, lu_inverse, lu_solve};
#[cfg(feature = "gpu")]
pub use crate::ops::linalg::{QrDecomposition, qr_decompose, qr_least_squares};
#[cfg(feature = "gpu")]
pub use crate::ops::linalg::{SvdDecomposition, svd_decompose, svd_pinv, svd_values};
#[cfg(feature = "gpu")]
pub use cholesky::cholesky_f64;
#[cfg(feature = "gpu")]
pub use eigh::eigh_f64;
#[cfg(feature = "gpu")]
pub use gen_eigh::{GenEighDecomposition, gen_eigh_f64, gen_eigh_identity_b};
pub use graph::{belief_propagation_chain, disordered_laplacian, effective_rank, graph_laplacian};
#[cfg(feature = "gpu")]
pub use solve::{solve_f64, solve_f64_cpu};
