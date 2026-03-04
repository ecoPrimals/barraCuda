// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparsity-based iterative surrogate sampling
//!
//! Implements the SparsitySampler algorithm from Diaw et al. (2024):
//! an iterative workflow that alternates between optimization (evaluation gathering)
//! and surrogate model training to achieve both exploitation and exploration.
//!
//! # Algorithm
//!
//! ```text
//! 1. Generate initial samples via maximin LHS
//! 2. Evaluate objective at initial samples
//! 3. LOOP until budget exhausted:
//!    a. Train RBF surrogate on ALL evaluations
//!    b. Use surrogate to identify promising regions (minimize predicted value)
//!    c. Run multi-start NM on the SURROGATE to find candidate points
//!    d. Evaluate TRUE objective at candidate points
//!    e. Add evaluations to cache
//! ```
//!
//! # Hybrid Evaluation Strategy
//!
//! For large datasets (n > 100), the distance matrix computation uses GPU
//! acceleration via `cdist.wgsl`. Enable with `SparsitySamplerConfig::with_gpu(device)`.
//!
//! # References
//!
//! - Diaw, A. et al. (2024). "Efficient learning of accurate surrogates for
//!   simulations of complex systems." Nature Machine Intelligence.

mod config;
mod filter;
mod result;
mod sampler;
mod sampler_gpu;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;

pub use config::{PenaltyFilter, SparsitySamplerConfig};
pub use result::{IterationResult, SparsitySamplerResult};
pub use sampler::sparsity_sampler;
pub use sampler_gpu::sparsity_sampler_gpu;
