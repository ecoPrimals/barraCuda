// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optimization algorithms for parameter fitting and model calibration
//!
//! This module provides both gradient-free and gradient-based optimization
//! methods suitable for scientific computing and machine learning.
//!
//! # Algorithms
//!
//! ## Optimization
//!
//! - **Nelder-Mead**: Simplex method for gradient-free local optimization
//! - **Multi-start Nelder-Mead**: Global optimization with LHS initial guesses
//! - **BFGS**: Quasi-Newton method with gradient information
//!
//! ## Root-Finding
//!
//! - **Bisection**: Reliable but slow O(n) convergence (CPU)
//! - **Batched Bisection GPU**: Parallel bisection for many problems (GPU)
//! - **Newton-Raphson**: Fast O(n²) convergence with derivatives
//! - **Brent's Method**: Best of both worlds - reliable and fast
//! - **Secant Method**: Newton-like without analytical derivatives
//!
//! ## Diagnostics & Utilities
//!
//! - **Evaluation Cache**: Record all evaluations for surrogate training
//! - **Convergence Diagnostics**: Detect stagnation, oscillation, divergence
//! - **Adaptive Penalty**: Data-driven penalty for constrained optimization
//!
//! # Cross-Domain Applications
//!
//! - **Physics**: Nuclear EOS parameter fitting, force-field calibration
//! - **ML**: Hyperparameter tuning, architecture search
//! - **Graphics**: Camera calibration, rendering parameter optimization
//! - **Audio**: Filter design, codec parameter tuning
//!
//! # Examples
//!
//! ```
//! use barracuda::optimize::{nelder_mead, multi_start_nelder_mead};
//!
//! // Local optimization
//! let rosenbrock = |x: &[f64]| {
//!     let (a, b) = (1.0, 100.0);
//!     (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
//! };
//!
//! let x0 = vec![0.0, 0.0];
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//!
//! let (x_best, f_best, n_evals) = nelder_mead(
//!     rosenbrock,
//!     &x0,
//!     &bounds,
//!     1000,
//!     1e-8,
//! )?;
//!
//! // Global optimization with evaluation recording
//! let (best, cache, _) = multi_start_nelder_mead(
//!     rosenbrock,
//!     &bounds,
//!     16,     // n_starts (like SparsitySampler npts=16)
//!     1000,   // max_iter per start
//!     1e-8,   // tolerance
//!     42,     // seed
//! )?;
//!
//! // Use cache for surrogate training
//! let (x_data, y_data) = cache.training_data();
//! # Ok::<(), barracuda::error::BarracudaError>(())
//! ```

pub mod batched_bisection_gpu; // EVOLVED: GPU batched root-finding (Feb 16, 2026)
pub mod batched_nelder_mead_gpu;
mod batched_nelder_mead_pipeline;
pub mod bfgs;
pub mod bisect;
pub mod brent;
pub mod brent_gpu;
pub mod diagnostics;
pub mod eval_record;
pub mod lbfgs;
pub mod lbfgs_gpu;
pub mod multi_start;
pub mod nelder_mead;
pub mod nelder_mead_gpu; // EVOLVED: GPU-resident optimizer (Feb 14, 2026)
pub mod newton;
pub mod penalty;
pub mod solver_state;

pub use batched_bisection_gpu::{BatchedBisectionGpu, BisectionResult};
pub use batched_nelder_mead_gpu::{
    BatchNelderMeadConfig, NelderMeadResult, batched_nelder_mead_gpu,
};
pub use bfgs::{BfgsConfig, BfgsResult, bfgs, bfgs_numerical, numerical_gradient};
pub use bisect::bisect;
pub use brent::{BrentResult, brent, brent_minimize};
pub use brent_gpu::{BrentFunction, BrentGpu, BrentGpuResult};
pub use diagnostics::{
    ConvergenceDiagnostics, ConvergenceState, convergence_diagnostics, should_stop_early,
};
pub use eval_record::{EvaluationCache, EvaluationRecord};
pub use lbfgs::{LbfgsConfig, LbfgsResult, lbfgs, lbfgs_numerical};
pub use lbfgs_gpu::{LbfgsGpu, LbfgsGpuConfig, LbfgsGpuResult};
pub use multi_start::{SolverResult, multi_start_nelder_mead};
pub use nelder_mead::nelder_mead;
pub use nelder_mead_gpu::{NelderMeadGpu, NelderMeadGpuResult}; // GPU-resident optimizer
pub use newton::{NewtonResult, newton, newton_numerical, secant};
pub use penalty::{AdaptivePenalty, PenaltyConfig, adaptive_penalty, adaptive_penalty_mad};
pub use solver_state::{ResumableNelderMead, SolverStatus};
