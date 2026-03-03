//! GPU path for SparsitySampler algorithm.

use crate::optimize::eval_record::EvaluationCache;
use crate::sample::latin_hypercube;
use crate::surrogate::adaptive::train_adaptive_gpu;
use crate::surrogate::RBFSurrogate;

use super::filter::compute_surrogate_rmse;
use super::result::{IterationResult, SparsitySamplerResult};
use super::sampler::run_nm_batch;
use super::SparsitySamplerConfig;
use crate::error::{BarracudaError, Result};

/// Run the SparsitySampler algorithm with GPU-accelerated surrogate training.
pub async fn sparsity_sampler_gpu<F>(
    f: F,
    bounds: &[(f64, f64)],
    config: &SparsitySamplerConfig,
) -> Result<SparsitySamplerResult>
where
    F: Fn(&[f64]) -> f64,
{
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

    // Phase 1: Initial sampling via LHS
    let initial_points = latin_hypercube(config.n_initial, bounds, config.seed)?;

    for point in &initial_points {
        let val = f(point);
        cache.record(point.clone(), val);
    }

    let mut last_surrogate = None;

    for iter in 0..config.n_iterations {
        let iter_start_evals = cache.len();

        let (x_data, y_data) = cache.training_data();

        let (surrogate, used_gpu) = if config.should_use_gpu(x_data.len()) {
            let device = config
                .gpu_device
                .as_ref()
                .ok_or_else(|| BarracudaError::InvalidInput {
                    message: "gpu_device must be set when should_use_gpu returns true; \
                              set SparsitySamplerConfig::gpu_device before calling \
                              sparsity_sampler_gpu"
                        .to_string(),
                })?
                .clone();
            match train_adaptive_gpu(&x_data, &y_data, config.kernel, config.smoothing, device)
                .await
            {
                Ok((s, _diag)) => (s, true),
                Err(_) => {
                    let dev = config
                        .gpu_device
                        .as_ref()
                        .ok_or_else(|| BarracudaError::InvalidInput {
                            message: "gpu_device required for fallback".to_string(),
                        })?
                        .clone();
                    match RBFSurrogate::train(
                        dev,
                        &x_data,
                        &y_data,
                        config.kernel,
                        config.smoothing,
                    ) {
                        Ok(s) => (s, false),
                        Err(_) => {
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
                    }
                }
            }
        } else {
            let dev = config
                .gpu_device
                .as_ref()
                .ok_or_else(|| BarracudaError::InvalidInput {
                    message: "gpu_device must be set for sparsity_sampler_gpu".to_string(),
                })?
                .clone();
            match RBFSurrogate::train(dev, &x_data, &y_data, config.kernel, config.smoothing) {
                Ok(s) => (s, false),
                Err(_) => {
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
            }
        };

        let surrogate_error = compute_surrogate_rmse(&surrogate, &x_data, &y_data);

        let surrogate_ref = &surrogate;
        let surrogate_objective = |x: &[f64]| {
            surrogate_ref
                .predict(&[x.to_vec()])
                .map(|v| v[0])
                .unwrap_or(f64::INFINITY)
        };

        let iter_seed = config.seed.wrapping_add((iter as u64 + 1) * 10007);
        let candidate_points = latin_hypercube(config.n_solvers, bounds, iter_seed)?;

        let mut iter_best_f = f64::INFINITY;
        let n_direct = config.n_direct_solvers.min(candidate_points.len());

        for (i, x0) in candidate_points.iter().enumerate() {
            if i < n_direct {
                let (x_star, f_star, _) = crate::optimize::nelder_mead(
                    &f,
                    x0,
                    bounds,
                    config.max_eval_per_solver,
                    config.tol,
                )?;
                cache.record(x_star, f_star);
                if f_star < iter_best_f {
                    iter_best_f = f_star;
                }
            } else {
                let (x_star, _, _) = crate::optimize::nelder_mead(
                    surrogate_objective,
                    x0,
                    bounds,
                    config.max_eval_per_solver,
                    config.tol,
                )?;
                let f_true = f(&x_star);
                cache.record(x_star, f_true);
                if f_true < iter_best_f {
                    iter_best_f = f_true;
                }
            }
        }

        iteration_results.push(IterationResult {
            iteration: iter,
            best_f: iter_best_f,
            n_new_evals: cache.len() - iter_start_evals,
            total_evals: cache.len(),
            surrogate_error: Some(surrogate_error),
            used_gpu,
        });

        last_surrogate = Some(surrogate);
    }

    let (x_best, f_best) = match cache.best() {
        Some(record) => (record.x.clone(), record.f),
        None => {
            return Err(BarracudaError::Internal(
                "No evaluations recorded".to_string(),
            ))
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
