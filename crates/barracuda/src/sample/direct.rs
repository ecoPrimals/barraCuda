// SPDX-License-Identifier: AGPL-3.0-or-later
//! Direct round-based optimization sampler
//!
//! Unlike surrogate-guided optimization (`SparsitySampler`), this module runs
//! multi-start Nelder-Mead directly on the true objective function. The surrogate
//! is trained for monitoring/quality assessment only, not for guiding optimization.
//!
//! # Algorithm (hotSpring validated)
//!
//! ```text
//! FOR round = 0..max_rounds:
//!   1. Generate starting points (LHS or warm-start seeds)
//!   2. Run multi-start NM on TRUE objective
//!   3. Add all evaluations to cache
//!   4. Train surrogate on filtered cache (monitoring only)
//!   5. Compute LOO-CV RMSE (quality metric)
//!   6. If no improvement for patience rounds → early stop
//! ```
//!
//! # When to Use
//!
//! - **Smooth landscapes**: Direct NM outperforms surrogate-guided on smooth objectives
//! - **L1 optimization**: SEMF-like models where the landscape is well-behaved
//! - **Validation**: When you want to compare surrogate-guided vs direct approaches
//!
//! # Reference
//!
//! hotSpring validation: `surrogate.rs::round_based_direct_optimization()`
//! Result: χ²/datum = 1.19 on L1 (82% better than scipy)

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::optimize::eval_record::EvaluationCache;
use crate::sample::latin_hypercube;
use crate::surrogate::{RBFKernel, RBFSurrogate, loo_cv_optimal_smoothing};
use std::sync::Arc;

/// Configuration for direct round-based optimization.
#[derive(Debug, Clone)]
pub struct DirectSamplerConfig {
    /// Number of rounds (default: 5)
    pub n_rounds: usize,
    /// NM solvers per round (default: 8)
    pub n_solvers: usize,
    /// Max evaluations per solver (default: 200)
    pub max_eval_per_solver: usize,
    /// NM convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// Early stopping patience (default: 2)
    pub patience: usize,
    /// Improvement threshold for early stopping (default: 1e-6)
    pub improvement_threshold: f64,
    /// Random seed
    pub seed: u64,
    /// Warm-start seeds (optional)
    pub warm_start_seeds: Vec<Vec<f64>>,
    /// RBF kernel for surrogate monitoring (default: `ThinPlateSpline`)
    pub kernel: RBFKernel,
    /// Enable auto-smoothing for surrogate (default: true)
    pub auto_smoothing: bool,
}

impl DirectSamplerConfig {
    /// Create a default configuration.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            n_rounds: 5,
            n_solvers: 8,
            max_eval_per_solver: 200,
            tol: 1e-8,
            patience: 2,
            improvement_threshold: 1e-6,
            seed,
            warm_start_seeds: Vec::new(),
            kernel: RBFKernel::ThinPlateSpline,
            auto_smoothing: true,
        }
    }

    /// Set number of optimization rounds.
    #[must_use]
    pub fn with_rounds(mut self, n: usize) -> Self {
        self.n_rounds = n;
        self
    }

    /// Set number of NM solvers per round.
    #[must_use]
    pub fn with_solvers(mut self, n: usize) -> Self {
        self.n_solvers = n;
        self
    }

    /// Set max evaluations per solver.
    #[must_use]
    pub fn with_eval_budget(mut self, n: usize) -> Self {
        self.max_eval_per_solver = n;
        self
    }

    /// Set NM convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set early stopping patience.
    #[must_use]
    pub fn with_patience(mut self, p: usize) -> Self {
        self.patience = p;
        self
    }

    /// Set warm-start seeds from previous optimization.
    #[must_use]
    pub fn with_warm_start(mut self, seeds: Vec<Vec<f64>>) -> Self {
        self.warm_start_seeds = seeds;
        self
    }

    /// Total evaluation budget (approximate).
    #[must_use]
    pub fn total_budget(&self) -> usize {
        self.n_rounds * self.n_solvers * self.max_eval_per_solver
    }
}

/// Result of direct optimization round.
#[derive(Debug, Clone)]
pub struct RoundResult {
    /// Round number (0-indexed)
    pub round: usize,
    /// Best f found in this round
    pub best_f: f64,
    /// Number of evaluations in this round
    pub n_evals: usize,
    /// Surrogate LOO-CV RMSE (monitoring metric)
    pub surrogate_rmse: Option<f64>,
    /// Improvement from previous round
    pub improvement: f64,
}

/// Result of direct round-based optimization.
#[derive(Debug)]
pub struct DirectSamplerResult {
    /// Best point found
    pub x_best: Vec<f64>,
    /// Best function value
    pub f_best: f64,
    /// All evaluations
    pub cache: EvaluationCache,
    /// Per-round diagnostics
    pub rounds: Vec<RoundResult>,
    /// Final surrogate (for monitoring/analysis)
    pub surrogate: Option<RBFSurrogate>,
    /// Was early stopped?
    pub early_stopped: bool,
}

impl DirectSamplerResult {
    /// Extract top-k points as warm-start seeds for subsequent optimization.
    ///
    /// Returns the k points with lowest function values, suitable for seeding
    /// another optimization round.
    #[must_use]
    pub fn top_k_seeds(&self, k: usize) -> Vec<Vec<f64>> {
        let mut records: Vec<_> = self.cache.records().to_vec();
        records.sort_by(|a, b| a.f.partial_cmp(&b.f).unwrap_or(std::cmp::Ordering::Equal));
        records.into_iter().take(k).map(|r| r.x).collect()
    }

    /// Get total number of true objective evaluations.
    #[must_use]
    pub fn total_evals(&self) -> usize {
        self.cache.len()
    }
}

/// Run direct round-based optimization.
///
/// Alternates between multi-start NM optimization and surrogate monitoring,
/// with early stopping based on improvement threshold.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `bounds` - Box bounds `[(min, max), ...]` for each dimension
/// * `config` - Sampler configuration
///
/// # Returns
///
/// [`DirectSamplerResult`] with best solution, all evaluations, and diagnostics.
///
/// # Example
///
/// ```ignore
/// use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
///
/// let config = DirectSamplerConfig::new(42)
///     .with_rounds(5)
///     .with_solvers(8)
///     .with_patience(2);
///
/// let result = direct_sampler(
///     |x| x.iter().map(|v| v * v).sum(),  // Sphere function
///     &[(-5.0, 5.0), (-5.0, 5.0)],
///     &config,
/// )?;
///
/// println!("Best: {:?} = {}", result.x_best, result.f_best);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if bounds is empty, or if multi-start NM or surrogate training fails.
pub fn direct_sampler<F>(
    device: Arc<WgpuDevice>,
    f: F,
    bounds: &[(f64, f64)],
    config: &DirectSamplerConfig,
) -> Result<DirectSamplerResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n_dims = bounds.len();
    if n_dims == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "bounds cannot be empty".to_string(),
        });
    }

    let mut cache = EvaluationCache::with_capacity(config.total_budget());
    let mut rounds = Vec::with_capacity(config.n_rounds);
    let mut last_surrogate = None;
    let mut best_f = f64::INFINITY;
    let mut no_improvement_count = 0;
    let mut early_stopped = false;

    // Evaluate warm-start seeds first
    for seed in &config.warm_start_seeds {
        if seed.len() == n_dims {
            let val = f(seed);
            cache.record(seed.clone(), val);
            if val < best_f {
                best_f = val;
            }
        }
    }

    for round in 0..config.n_rounds {
        let round_start_evals = cache.len();
        let prev_best = best_f;

        // Generate starting points for this round
        let round_seed = config.seed.wrapping_add((round as u64 + 1) * 10007);
        let starts = latin_hypercube(config.n_solvers, bounds, round_seed)?;

        // Run multi-start NM on TRUE objective
        for x0 in &starts {
            let (x_opt, f_opt, _) = crate::optimize::nelder_mead(
                &f,
                x0,
                bounds,
                config.max_eval_per_solver,
                config.tol,
            )?;

            cache.record(x_opt, f_opt);
            if f_opt < best_f {
                best_f = f_opt;
            }
        }

        // Train surrogate for monitoring (not guiding)
        let (x_data, y_data) = cache.training_data();
        let surrogate_rmse = if x_data.len() >= 2 {
            // Auto-smoothing via LOO-CV
            let smoothing = if config.auto_smoothing {
                loo_cv_optimal_smoothing(device.clone(), &x_data, &y_data, config.kernel, None)
                    .map(|r| r.smoothing)
                    .unwrap_or(1e-6)
            } else {
                1e-6
            };

            match RBFSurrogate::train(device.clone(), &x_data, &y_data, config.kernel, smoothing) {
                Ok(surr) => {
                    let rmse = surr.loo_cv_rmse().ok();
                    last_surrogate = Some(surr);
                    rmse
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Check improvement
        let improvement = prev_best - best_f;
        if improvement < config.improvement_threshold {
            no_improvement_count += 1;
        } else {
            no_improvement_count = 0;
        }

        rounds.push(RoundResult {
            round,
            best_f,
            n_evals: cache.len() - round_start_evals,
            surrogate_rmse,
            improvement,
        });

        // Early stopping
        if no_improvement_count >= config.patience {
            early_stopped = true;
            break;
        }
    }

    // Extract best overall result
    let (x_best, f_best) = match cache.best() {
        Some(record) => (record.x.clone(), record.f),
        None => {
            return Err(BarracudaError::Internal(
                "No evaluations recorded".to_string(),
            ));
        }
    };

    Ok(DirectSamplerResult {
        x_best,
        f_best,
        cache,
        rounds,
        surrogate: last_surrogate,
        early_stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available_sync;

    #[test]
    fn test_direct_sampler_sphere() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let config = DirectSamplerConfig::new(42)
            .with_rounds(3)
            .with_solvers(4)
            .with_eval_budget(100);
        let result = direct_sampler(device, sphere, &bounds, &config).unwrap();

        // Should find near-zero minimum
        assert!(
            result.f_best < 0.01,
            "Failed to minimize: {}",
            result.f_best
        );
        assert_eq!(result.x_best.len(), 2);
    }

    #[test]
    fn test_direct_sampler_with_warm_start() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        // Warm start near the optimum
        let config = DirectSamplerConfig::new(42)
            .with_rounds(2)
            .with_solvers(2)
            .with_warm_start(vec![vec![0.1, 0.1]]);

        let result = direct_sampler(device, sphere, &bounds, &config).unwrap();

        // Should find very good solution with warm start
        assert!(result.f_best < 0.001);
    }

    #[test]
    fn test_direct_sampler_early_stop() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let constant = |_: &[f64]| 1.0;
        let bounds = vec![(-1.0, 1.0)];

        let config = DirectSamplerConfig::new(42)
            .with_rounds(10)
            .with_patience(2);

        let result = direct_sampler(device, constant, &bounds, &config).unwrap();

        // Should early stop
        assert!(result.early_stopped);
        assert!(result.rounds.len() < 10);
    }

    #[test]
    fn test_direct_sampler_diagnostics() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let sphere = |x: &[f64]| x.iter().map(|v| v * v).sum();
        let bounds = vec![(-5.0, 5.0)];
        let config = DirectSamplerConfig::new(42).with_rounds(3).with_solvers(2);
        let result = direct_sampler(device, sphere, &bounds, &config).unwrap();

        // Check round diagnostics
        assert!(!result.rounds.is_empty());
        for (i, round) in result.rounds.iter().enumerate() {
            assert_eq!(round.round, i);
            assert!(round.n_evals > 0);
        }
    }
}
