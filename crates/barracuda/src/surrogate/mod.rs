// SPDX-License-Identifier: AGPL-3.0-only
//! Surrogate modeling for expensive function approximation
//!
//! This module provides radial basis function (RBF) surrogates for
//! approximating expensive-to-evaluate functions. Common use cases:
//! - Black-box optimization with expensive simulations
//! - Emulation of physics models
//! - Response surface methodology
//!
//! # Dual-Precision Architecture
//!
//! The RBF surrogate uses `BarraCuda`'s dual-precision pattern:
//! 1. **GPU (f32)**: Compute pairwise distances via cdist shader (O(n²) bottleneck)
//! 2. **CPU (f64)**: Apply kernel, assemble matrix, solve for weights
//!
//! This gives 14× speedup on training while maintaining f64 precision
//! for numerically sensitive kernel evaluation and linear algebra.
//!
//! # Examples
//!
//! ```no_run
//! use barracuda::surrogate::{RBFSurrogate, RBFKernel};
//! use barracuda::prelude::WgpuDevice;
//! use std::sync::Arc;
//!
//! # async fn example() -> barracuda::error::Result<()> {
//! let device = Arc::new(WgpuDevice::new().await?);
//! // Training data: y = x²
//! let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
//! let y_train = vec![0.0, 1.0, 4.0, 9.0];
//!
//! // Train surrogate
//! let surrogate = RBFSurrogate::train(
//!     device,
//!     &x_train,
//!     &y_train,
//!     RBFKernel::ThinPlateSpline,
//!     1e-12,
//! )?;
//!
//! // Predict at new points
//! let x_eval = vec![vec![1.5], vec![2.5]];
//! let y_pred = surrogate.predict(&x_eval)?;
//!
//! assert!((y_pred[0] - 2.25).abs() < 0.1);  // ≈ 1.5²
//! assert!((y_pred[1] - 6.25).abs() < 0.1);  // ≈ 2.5²
//! # Ok(())
//! # }
//! ```

pub mod adaptive;
pub mod kernels;
pub mod rbf;

pub use kernels::RBFKernel;
pub use rbf::{LooSmoothing, RBFSurrogate, loo_cv_optimal_smoothing};
