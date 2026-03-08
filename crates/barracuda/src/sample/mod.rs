// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sampling strategies for parameter space exploration
//!
//! Provides space-filling and guided sampling methods used across domains:
//! - **Surrogate learning**: Train RBF models on well-distributed points
//! - **Design of experiments**: Explore parameter spaces efficiently
//! - **Neural architecture search**: Evaluate hyperparameter configurations
//! - **Monte Carlo**: Generate quasi-random initial conditions
//!
//! # Available Samplers
//!
//! - [`latin_hypercube`] — Space-filling design with stratified intervals
//! - [`random_uniform`] — Uniform random sampling within bounds
//! - [`maximin::maximin_lhs`] — Maximin-optimized LHS (maximize min pairwise distance)
//! - [`sparsity::sparsity_sampler`] — Iterative surrogate-directed sampling (Diaw et al. 2024)
//! - [`direct::direct_sampler`] — Round-based direct NM on true objective (hotSpring)
//! - [`metropolis::boltzmann_sampling`] — Metropolis-Hastings MCMC with Boltzmann acceptance
//! - [`sobol::sobol_sequence`] — Low-discrepancy quasi-random sequences
//!
//! # Examples
//!
//! ```
//! use barracuda::sample::{latin_hypercube, random_uniform};
//!
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//!
//! // LHS: one sample per row/column in hypercube
//! let lhs_points = latin_hypercube(100, &bounds, 42)?;
//! assert_eq!(lhs_points.len(), 100);
//! assert_eq!(lhs_points[0].len(), 2);
//!
//! // Random: uniform within bounds
//! let rng_points = random_uniform(100, &bounds, 42);
//! assert_eq!(rng_points.len(), 100);
//! # Ok::<(), barracuda::error::BarracudaError>(())
//! ```
//!
//! ```
//! use barracuda::sample::sobol::sobol_scaled;
//!
//! let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
//! let points = sobol_scaled(100, &bounds).unwrap();
//! // Points are more uniformly distributed than pseudo-random
//! ```

#[cfg(feature = "gpu")]
pub mod direct;
pub mod lhs;
pub mod maximin;
pub mod metropolis;
pub mod sobol;
#[cfg(feature = "gpu")]
pub mod sparsity;

#[cfg(feature = "gpu")]
/// Latin hypercube sampling shader.
pub static WGSL_LHS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_str!(
        "../shaders/sample/lhs_f64.wgsl"
    ).to_string()
});

#[cfg(feature = "gpu")]
/// f64 is the canonical source — math is universal, precision is silicon.
static WGSL_METROPOLIS_F64: &str = include_str!("../shaders/sample/metropolis_f64.wgsl");
#[cfg(feature = "gpu")]
/// WGSL shader: parallel Metropolis-Hastings MCMC
/// WGSL kernel for Metropolis-Hastings sampling.
pub static WGSL_METROPOLIS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    WGSL_METROPOLIS_F64.to_string()
});

#[cfg(feature = "gpu")]
pub use direct::{DirectSamplerConfig, DirectSamplerResult, direct_sampler};
pub use lhs::{latin_hypercube, random_uniform};
pub use metropolis::{BoltzmannResult, boltzmann_sampling};
pub use sobol::{SobolGenerator, sobol_scaled, sobol_sequence};
#[cfg(feature = "gpu")]
pub use sparsity::{PenaltyFilter, SparsitySamplerConfig};
