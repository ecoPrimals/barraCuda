// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU path for `SparsitySampler` algorithm.

use crate::device::WgpuDevice;
use crate::optimize::eval_record::EvaluationCache;
use crate::optimize::multi_start::SolverResult;
use crate::sample::latin_hypercube;
use crate::surrogate::{RBFSurrogate, loo_cv_optimal_smoothing};
use std::sync::Arc;

use super::SparsitySamplerConfig;
use super::filter::compute_surrogate_rmse;
use super::filter::filter_training_data;
use super::result::{IterationResult, SparsitySamplerResult};
use crate::error::{BarracudaError, Result};

/// Run a batch of NM solvers on the true objective (fallback when surrogate fails).
///
/// Solvers run in parallel via rayon when the objective is `Send + Sync`.
pub(crate) fn run_nm_batch<F>(
    f: &F,
    bounds: &[(f64, f64)],
    config: &SparsitySamplerConfig,
    iter: usize,
    cache: &mut EvaluationCache,
) -> Result<SolverResult>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    use rayon::prelude::*;

    let seed = config.seed.wrapping_add((iter as u64 + 1) * 10007);
    let points = latin_hypercube(config.n_solvers, bounds, seed)?;

    let results: Vec<_> = points
        .par_iter()
        .map(|x0| {
            crate::optimize::nelder_mead(f, x0, bounds, config.max_eval_per_solver, config.tol)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut best_x = vec![0.0; bounds.len()];
    let mut best_f = f64::INFINITY;

    for (x_star, f_star, _) in results {
        cache.record(x_star.clone(), f_star);
        if f_star < best_f {
            best_f = f_star;
            best_x = x_star;
        }
    }

    Ok(SolverResult {
        x_best: best_x,
        f_best: best_f,
        n_evals: config.n_solvers * config.max_eval_per_solver,
        converged: false,
    })
}

/// Run the `SparsitySampler` algorithm (CPU path).
///
/// # Errors
///
/// Returns [`Err`] if bounds is empty, `n_initial` < 2, or if LHS/surrogate/NM fails.
pub fn sparsity_sampler<F>(
    device: Arc<WgpuDevice>,
    f: F,
    bounds: &[(f64, f64)],
    config: &SparsitySamplerConfig,
) -> Result<SparsitySamplerResult>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    use rayon::prelude::*;
    if bounds.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "bounds must be non-empty".to_string(),
        });
    }

    if config.n_initial < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "n_initial must be >= 2 for surrogate training".to_string(),
        });
    }

    let _n_dims = bounds.len();
    let mut cache = EvaluationCache::with_capacity(config.total_budget());
    let mut iteration_results = Vec::with_capacity(config.n_iterations);
    let mut current_smoothing = config.smoothing;

    // Phase 1: Initial sampling via LHS
    let initial_points = latin_hypercube(config.n_initial, bounds, config.seed)?;

    for point in &initial_points {
        let val = f(point);
        cache.record(point.clone(), val);
    }

    // Evaluate warm-start seeds (L1→L2 seeding pattern)
    for seed in &config.warm_start_seeds {
        if seed.len() == bounds.len() {
            let val = f(seed);
            cache.record(seed.clone(), val);
        }
    }

    // Iterative refinement loop
    let mut last_surrogate = None;

    for iter in 0..config.n_iterations {
        let iter_start_evals = cache.len();

        // Get training data from cache
        let (x_raw, y_raw) = cache.training_data();

        // Apply penalty filtering before surrogate training
        let (x_data, y_data) = filter_training_data(&x_raw, &y_raw, config.penalty_filter);

        // Skip if filtering removed too many points
        if x_data.len() < 2 {
            let nm_result = run_nm_batch(&f, bounds, config, iter, &mut cache)?;
            iteration_results.push(IterationResult {
                iteration: iter,
                best_f: nm_result.f_best,
                n_new_evals: cache.len() - iter_start_evals,
                total_evals: cache.len(),
                surrogate_error: None,
                used_gpu: false,
            });
            continue;
        }

        // Auto-smoothing via LOO-CV grid search (if enabled)
        if config.auto_smoothing {
            if let Ok(result) =
                loo_cv_optimal_smoothing(device.clone(), &x_data, &y_data, config.kernel, None)
            {
                current_smoothing = result.smoothing;
            }
        }

        let Ok(surrogate) = RBFSurrogate::train(
            device.clone(),
            &x_data,
            &y_data,
            config.kernel,
            current_smoothing,
        ) else {
            let nm_result = run_nm_batch(&f, bounds, config, iter, &mut cache)?;
            iteration_results.push(IterationResult {
                iteration: iter,
                best_f: nm_result.f_best,
                n_new_evals: cache.len() - iter_start_evals,
                total_evals: cache.len(),
                surrogate_error: None,
                used_gpu: false,
            });
            continue;
        };

        let surrogate_error = surrogate
            .loo_cv_rmse()
            .unwrap_or_else(|_| compute_surrogate_rmse(&surrogate, &x_data, &y_data));

        let surrogate_ref = &surrogate;
        let surrogate_objective = |x: &[f64]| {
            surrogate_ref
                .predict(&[x.to_vec()])
                .map(|v| v[0])
                .unwrap_or(f64::INFINITY)
        };

        let iter_seed = config.seed.wrapping_add((iter as u64 + 1) * 10007);
        let candidate_points = latin_hypercube(config.n_solvers, bounds, iter_seed)?;

        let n_direct = config.n_direct_solvers.min(candidate_points.len());

        // Run direct and surrogate-guided solvers in parallel via rayon.
        let solver_results: Vec<_> = candidate_points
            .par_iter()
            .enumerate()
            .map(|(i, x0)| {
                if i < n_direct {
                    let (x_star, f_star, evals) = crate::optimize::nelder_mead(
                        &f,
                        x0,
                        bounds,
                        config.max_eval_per_solver,
                        config.tol,
                    )?;
                    Ok((x_star, f_star, evals))
                } else {
                    let (x_star, _, evals) = crate::optimize::nelder_mead(
                        surrogate_objective,
                        x0,
                        bounds,
                        config.max_eval_per_solver,
                        config.tol,
                    )?;
                    let f_true = f(&x_star);
                    Ok((x_star, f_true, evals))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let mut iter_best_f = f64::INFINITY;
        for (x_star, f_star, _) in solver_results {
            cache.record(x_star, f_star);
            if f_star < iter_best_f {
                iter_best_f = f_star;
            }
        }

        iteration_results.push(IterationResult {
            iteration: iter,
            best_f: iter_best_f,
            n_new_evals: cache.len() - iter_start_evals,
            total_evals: cache.len(),
            surrogate_error: Some(surrogate_error),
            used_gpu: false,
        });

        last_surrogate = Some(surrogate);
    }

    let (x_best, f_best) = match cache.best() {
        Some(record) => (record.x.clone(), record.f),
        None => {
            return Err(BarracudaError::Internal(
                "No evaluations recorded".to_string(),
            ));
        }
    };

    Ok(SparsitySamplerResult {
        x_best,
        f_best,
        cache,
        surrogate: last_surrogate,
        iteration_results,
    })
}
