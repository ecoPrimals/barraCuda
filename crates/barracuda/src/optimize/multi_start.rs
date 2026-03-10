// SPDX-License-Identifier: AGPL-3.0-only
//! Multi-start Nelder-Mead optimization with evaluation recording
//!
//! Runs Nelder-Mead simplex from multiple initial guesses (generated via
//! Latin Hypercube Sampling) to explore parameter space globally. Captures
//! ALL function evaluations for surrogate training.
//!
//! # Algorithm
//!
//! 1. Generate `n_starts` initial guesses using Latin Hypercube Sampling
//! 2. Run Nelder-Mead from each initial guess
//! 3. Record every function evaluation (not just best results)
//! 4. Return the globally best solution plus the full evaluation cache
//!
//! This is the core of the `SparsitySampler` approach from Diaw et al. (2024):
//! multiple parallel solvers exploring different regions produce space-filling
//! evaluations that are both exploitative (near optima) and exploratory (initial
//! phases), making them ideal training data for RBF surrogates.
//!
//! # Cross-Domain Applications
//!
//! - **Nuclear EOS fitting**: 10D Skyrme parameter optimization
//! - **ML hyperparameter tuning**: Global search across learning rates, architectures
//! - **Materials science**: Force-field parameterization
//! - **Audio**: Filter design optimization
//!
//! # References
//!
//! - Diaw et al. (2024). "Efficient learning of accurate surrogates for simulations
//!   of complex systems." Nature Machine Intelligence.
//! - hotSpring: `control/surrogate/scripts/full_iterative_workflow.py`

use crate::error::{BarracudaError, Result};
use crate::optimize::eval_record::EvaluationCache;
use crate::sample::latin_hypercube;

/// Result of a single Nelder-Mead run within multi-start optimization.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Best point found by this solver
    pub x_best: Vec<f64>,
    /// Best function value found
    pub f_best: f64,
    /// Number of function evaluations used
    pub n_evals: usize,
    /// Whether the solver converged (met tolerance)
    pub converged: bool,
}

/// Run Nelder-Mead from multiple starting points with full evaluation recording.
///
/// This is the pure Rust equivalent of mystic's `SparsitySampler` strategy:
/// multiple parallel simplex solvers exploring different parameter space regions,
/// with ALL evaluations captured for surrogate model training.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `bounds` - Box bounds `[(min, max), ...]` for each dimension
/// * `n_starts` - Number of independent Nelder-Mead runs
/// * `max_iter_per_start` - Maximum function evaluations per run
/// * `tol` - Convergence tolerance (simplex standard deviation)
/// * `seed` - Random seed for initial point generation (LHS)
///
/// # Returns
///
/// Tuple of:
/// - `SolverResult` for the globally best solution
/// - `EvaluationCache` containing ALL evaluations from all solvers
/// - `Vec<SolverResult>` for each individual solver run
///
/// # Examples
///
/// ```
/// use barracuda::optimize::multi_start_nelder_mead;
///
/// // Rastrigin function (many local minima, global min at origin)
/// let rastrigin = |x: &[f64]| {
///     let n = x.len() as f64;
///     10.0 * n + x.iter().map(|&xi| xi.powi(2) - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
/// };
///
/// let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
///
/// let (best, cache, all_results) = multi_start_nelder_mead(
///     rastrigin,
///     &bounds,
///     16,    // n_starts (like SparsitySampler npts=16)
///     500,   // max_iter per start
///     1e-8,  // tolerance
///     42,    // seed
/// )?;
///
/// // Should find a point near the global minimum
/// println!("Best: f={:.4}, x={:?}", best.f_best, best.x_best);
///
/// // All evaluations captured for surrogate training
/// println!("Total evaluations: {}", cache.len());
/// assert!(cache.len() > 16); // More evals than starts
///
/// // Individual solver results available
/// assert_eq!(all_results.len(), 16);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # Errors
///
/// Returns [`Err`] if `n_starts` is 0, or if LHS or Nelder-Mead fails.
pub fn multi_start_nelder_mead<F>(
    f: F,
    bounds: &[(f64, f64)],
    n_starts: usize,
    max_iter_per_start: usize,
    tol: f64,
    seed: u64,
) -> Result<(SolverResult, EvaluationCache, Vec<SolverResult>)>
where
    F: Fn(&[f64]) -> f64,
{
    if n_starts == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "n_starts must be > 0".to_string(),
        });
    }

    if bounds.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "bounds must be non-empty".to_string(),
        });
    }

    let n_dims = bounds.len();

    // Generate initial guesses via Latin Hypercube Sampling
    let initial_points = latin_hypercube(n_starts, bounds, seed)?;

    // Global evaluation cache captures ALL evaluations
    let mut global_cache = EvaluationCache::with_capacity(n_starts * max_iter_per_start);
    let mut all_results = Vec::with_capacity(n_starts);

    // Run Nelder-Mead from each starting point
    for x0 in &initial_points {
        let result = run_single_nm(
            &f,
            x0,
            bounds,
            n_dims,
            max_iter_per_start,
            tol,
            &mut global_cache,
        );
        all_results.push(result);
    }

    // Find global best across all runs
    let best_idx = all_results
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.f_best
                .partial_cmp(&b.f_best)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(0, |(idx, _)| idx);

    let best = all_results[best_idx].clone();

    Ok((best, global_cache, all_results))
}

/// Run a single Nelder-Mead instance, recording ALL evaluations to the cache.
///
/// This is an instrumented version of the Nelder-Mead algorithm that captures
/// every function evaluation for surrogate training.
fn run_single_nm<F>(
    f: &F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    n: usize,
    max_iter: usize,
    tol: f64,
    cache: &mut EvaluationCache,
) -> SolverResult
where
    F: Fn(&[f64]) -> f64,
{
    // Nelder-Mead parameters
    const ALPHA: f64 = 1.0; // Reflection
    const GAMMA: f64 = 2.0; // Expansion
    const RHO: f64 = 0.5; // Contraction
    const SIGMA: f64 = 0.5; // Shrinkage

    // Instrumented evaluation: records to cache
    let eval = |x: &[f64], cache: &mut EvaluationCache| -> f64 {
        let val = f(x);
        cache.record(x.to_vec(), val);
        val
    };

    // Initialize simplex
    let mut simplex = Vec::with_capacity(n + 1);
    let mut f_vals = Vec::with_capacity(n + 1);

    let x0_bounded = project_bounds(x0, bounds);
    let f0 = eval(&x0_bounded, cache);
    simplex.push(x0_bounded);
    f_vals.push(f0);

    for i in 0..n {
        let mut x = x0.to_vec();
        let delta = 0.05 * (bounds[i].1 - bounds[i].0).max(0.1);
        x[i] += delta;
        let x_bounded = project_bounds(&x, bounds);
        let fx = eval(&x_bounded, cache);
        simplex.push(x_bounded);
        f_vals.push(fx);
    }

    let mut n_evals = n + 1;
    let mut converged = false;

    for _iter in 0..max_iter {
        if n_evals >= max_iter {
            break;
        }

        // Sort by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&i, &j| {
            f_vals[i]
                .partial_cmp(&f_vals[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Convergence check
        let f_mean: f64 = f_vals.iter().sum::<f64>() / (n + 1) as f64;
        let f_std =
            (f_vals.iter().map(|&fi| (fi - f_mean).powi(2)).sum::<f64>() / (n + 1) as f64).sqrt();

        if f_std < tol {
            converged = true;
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let x_reflect = reflect(&simplex[worst_idx], &centroid, ALPHA);
        let x_reflect = project_bounds(&x_reflect, bounds);
        let f_reflect = eval(&x_reflect, cache);
        n_evals += 1;

        if f_reflect < f_vals[best_idx] {
            // Expansion
            let x_expand = reflect(&simplex[worst_idx], &centroid, GAMMA);
            let x_expand = project_bounds(&x_expand, bounds);
            let f_expand = eval(&x_expand, cache);
            n_evals += 1;

            if f_expand < f_reflect {
                simplex[worst_idx] = x_expand;
                f_vals[worst_idx] = f_expand;
            } else {
                simplex[worst_idx] = x_reflect;
                f_vals[worst_idx] = f_reflect;
            }
        } else if f_reflect < f_vals[second_worst_idx] {
            simplex[worst_idx] = x_reflect;
            f_vals[worst_idx] = f_reflect;
        } else {
            // Contraction
            let x_contract = if f_reflect < f_vals[worst_idx] {
                reflect(&simplex[worst_idx], &centroid, ALPHA * RHO)
            } else {
                reflect(&simplex[worst_idx], &centroid, -RHO)
            };
            let x_contract = project_bounds(&x_contract, bounds);
            let f_contract = eval(&x_contract, cache);
            n_evals += 1;

            if f_contract < f_vals[worst_idx] {
                simplex[worst_idx] = x_contract;
                f_vals[worst_idx] = f_contract;
            } else {
                // Shrinkage
                for i in 0..=n {
                    if i != best_idx {
                        for j in 0..n {
                            simplex[i][j] = simplex[best_idx][j]
                                + SIGMA * (simplex[i][j] - simplex[best_idx][j]);
                        }
                        simplex[i] = project_bounds(&simplex[i], bounds);
                        f_vals[i] = eval(&simplex[i], cache);
                        n_evals += 1;
                    }
                }
            }
        }
    }

    // Return best from this run
    let best_idx = f_vals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx);

    SolverResult {
        x_best: simplex[best_idx].clone(),
        f_best: f_vals[best_idx],
        n_evals,
        converged,
    }
}

/// Reflect point x through centroid by factor alpha
fn reflect(x: &[f64], centroid: &[f64], alpha: f64) -> Vec<f64> {
    centroid
        .iter()
        .zip(x.iter())
        .map(|(&c, &xi)| c + alpha * (c - xi))
        .collect()
}

/// Project point onto box constraints
fn project_bounds(x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    x.iter()
        .zip(bounds.iter())
        .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_start_quadratic() {
        // Simple quadratic: minimum at (2, 3)
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        let (best, cache, results) = multi_start_nelder_mead(f, &bounds, 5, 500, 1e-8, 42).unwrap();

        assert!((best.x_best[0] - 2.0).abs() < 1e-3);
        assert!((best.x_best[1] - 3.0).abs() < 1e-3);
        assert!(best.f_best < 1e-5);
        assert_eq!(results.len(), 5);
        assert!(cache.len() > 5); // More evals than starts
    }

    #[test]
    fn test_multi_start_rosenbrock() {
        // Rosenbrock: global minimum at (1, 1)
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let (best, cache, _) = multi_start_nelder_mead(f, &bounds, 10, 2000, 1e-8, 42).unwrap();

        assert!((best.x_best[0] - 1.0).abs() < 0.1);
        assert!((best.x_best[1] - 1.0).abs() < 0.1);
        assert!(best.f_best < 0.01);
        assert!(cache.len() > 100); // Many evaluations captured
    }

    #[test]
    fn test_multi_start_rastrigin() {
        // Rastrigin: many local minima, global min at (0, 0) with f=0
        let rastrigin = |x: &[f64]| {
            let n = x.len() as f64;
            10.0 * n
                + x.iter()
                    .map(|&xi| xi.powi(2) - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<f64>()
        };

        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];

        let (best, cache, results) =
            multi_start_nelder_mead(rastrigin, &bounds, 16, 500, 1e-8, 42).unwrap();

        // Should find a decent minimum (not guaranteed global for Rastrigin)
        // but should be much better than a single start
        assert!(best.f_best < 5.0); // Good solution found
        assert_eq!(results.len(), 16);
        assert!(cache.len() > 100);
    }

    #[test]
    fn test_multi_start_captures_all_evals() {
        // Verify every evaluation is captured
        let f = |x: &[f64]| x[0].powi(2);
        let bounds = vec![(-10.0, 10.0)];

        let (_, cache, results) = multi_start_nelder_mead(f, &bounds, 3, 50, 1e-8, 42).unwrap();

        // Total evals in cache should equal sum of individual evals
        let total_individual: usize = results.iter().map(|r| r.n_evals).sum();
        assert_eq!(cache.len(), total_individual);
    }

    #[test]
    fn test_multi_start_training_data() {
        // Verify training data extraction works
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let (_, cache, _) = multi_start_nelder_mead(f, &bounds, 5, 100, 1e-8, 42).unwrap();

        let (x_data, y_data) = cache.training_data();
        assert_eq!(x_data.len(), y_data.len());
        assert_eq!(x_data.len(), cache.len());

        // Verify consistency: f(x) == y for all records
        for (xi, &yi) in x_data.iter().zip(y_data.iter()) {
            let expected = xi[0].powi(2) + xi[1].powi(2);
            assert!(
                (expected - yi).abs() < 1e-10,
                "Inconsistent: f({xi:?}) = {expected} != {yi}"
            );
        }
    }

    #[test]
    fn test_multi_start_high_dimensional() {
        // 5D optimization to test scalability
        let f = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(i, &xi)| (xi - i as f64).powi(2))
                .sum::<f64>()
        };
        let bounds: Vec<(f64, f64)> = (0..5).map(|_| (-10.0, 10.0)).collect();

        let (best, cache, results) =
            multi_start_nelder_mead(f, &bounds, 10, 1000, 1e-6, 42).unwrap();

        // Should find minimum near (0, 1, 2, 3, 4)
        assert!(best.f_best < 1.0);
        assert_eq!(results.len(), 10);
        assert!(cache.len() > 50);
    }

    #[test]
    fn test_multi_start_errors() {
        let f = |x: &[f64]| x[0].powi(2);
        let bounds = vec![(-1.0, 1.0)];

        // Zero starts
        assert!(multi_start_nelder_mead(f, &bounds, 0, 100, 1e-8, 42).is_err());

        // Empty bounds
        assert!(multi_start_nelder_mead(f, &[], 5, 100, 1e-8, 42).is_err());
    }

    #[test]
    fn test_multi_start_convergence_info() {
        // At least some solvers should converge on a simple problem
        let f = |x: &[f64]| (x[0] - 1.0).powi(2);
        let bounds = vec![(-10.0, 10.0)];

        let (_, _, results) = multi_start_nelder_mead(f, &bounds, 5, 500, 1e-6, 42).unwrap();

        let n_converged = results.iter().filter(|r| r.converged).count();
        assert!(n_converged > 0, "At least one solver should converge");
    }

    #[test]
    fn test_multi_start_global_vs_local() {
        // Function with two minima: f(x) = min at x=1 (local, f=0) and x=-3 (global, f=-1)
        let f = |x: &[f64]| {
            let x0 = x[0];
            // Two wells: one at x=1 (depth 0) and one at x=-3 (depth -1)
            let well1 = (x0 - 1.0).powi(2);
            let well2 = (x0 + 3.0).powi(2) - 1.0;
            well1.min(well2)
        };

        let bounds = vec![(-5.0, 5.0)];

        let (best, _, _) = multi_start_nelder_mead(f, &bounds, 10, 500, 1e-8, 42).unwrap();

        // Multi-start should find the global minimum (at x=-3, f=-1)
        assert!(best.f_best < 0.0, "Should find the deeper well (f < 0)");
    }
}
