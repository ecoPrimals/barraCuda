// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nelder-Mead simplex optimization algorithm

use crate::error::{BarracudaError, Result};

/// Minimize f(x) using Nelder-Mead simplex method with bounds
///
/// This is a gradient-free local optimization method suitable for
/// expensive black-box functions. It works by maintaining a simplex
/// of n+1 points and iteratively reflecting, expanding, and contracting
/// the simplex to find a minimum.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `x0` - Initial guess (length n)
/// * `bounds` - Box bounds [(min, max), ...] for each dimension
/// * `max_iter` - Maximum number of function evaluations
/// * `tol` - Convergence tolerance (simplex standard deviation)
///
/// # Returns
///
/// Tuple of (`x_best`, `f_best`, `n_evaluations`)
///
/// # Algorithm
///
/// Standard Nelder-Mead with:
/// - **Reflection** (α = 1.0)
/// - **Expansion** (γ = 2.0)
/// - **Contraction** (ρ = 0.5)
/// - **Shrinkage** (σ = 0.5)
///
/// Plus box constraints enforcement via projection.
///
/// # Examples
///
/// ```
/// use barracuda::optimize::nelder_mead;
///
/// // Minimize Rosenbrock function
/// let rosenbrock = |x: &[f64]| {
///     let (a, b) = (1.0, 100.0);
///     (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
/// };
///
/// let x0 = vec![0.0, 0.0];
/// let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
///
/// let (x_best, f_best, n_evals) = nelder_mead(
///     rosenbrock,
///     &x0,
///     &bounds,
///     1000,
///     1e-8,
/// )?;
///
/// // Should find minimum at (1, 1)
/// assert!((x_best[0] - 1.0).abs() < 1e-3);
/// assert!((x_best[1] - 1.0).abs() < 1e-3);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # References
///
/// - Nelder, J. A.; Mead, R. (1965). "A simplex method for function minimization"
/// - Numerical Recipes, 3rd Edition, Section 10.5
/// - scipy.optimize.fmin
///
/// # Errors
///
/// Returns [`Err`] if bounds length does not match x0 length.
pub fn nelder_mead<F>(
    f: F,
    x0: &[f64],
    bounds: &[(f64, f64)],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, f64, usize)>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();

    if bounds.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Bounds length {} must match x0 length {}", bounds.len(), n),
        });
    }

    // Nelder-Mead parameters
    const ALPHA: f64 = 1.0; // Reflection
    const GAMMA: f64 = 2.0; // Expansion
    const RHO: f64 = 0.5; // Contraction
    const SIGMA: f64 = 0.5; // Shrinkage

    // Initialize simplex: n+1 points
    let mut simplex = Vec::with_capacity(n + 1);
    let mut f_vals = Vec::with_capacity(n + 1);

    // First vertex: x0
    let x0_bounded = project_bounds(x0, bounds);
    let f0 = f(&x0_bounded);
    simplex.push(x0_bounded);
    f_vals.push(f0);

    // Generate remaining n vertices by perturbing each coordinate
    for i in 0..n {
        let mut x = x0.to_vec();
        let delta = 0.05 * (bounds[i].1 - bounds[i].0).max(0.1);
        x[i] += delta;

        let x_bounded = project_bounds(&x, bounds);
        let fx = f(&x_bounded);
        simplex.push(x_bounded);
        f_vals.push(fx);
    }

    let mut n_evals = n + 1;

    // Main loop
    for _iter in 0..max_iter {
        // Sort simplex by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&i, &j| {
            f_vals[i]
                .partial_cmp(&f_vals[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence: standard deviation of function values
        let f_mean: f64 = f_vals.iter().sum::<f64>() / (n + 1) as f64;
        let f_std =
            (f_vals.iter().map(|&fi| (fi - f_mean).powi(2)).sum::<f64>() / (n + 1) as f64).sqrt();

        if f_std < tol {
            return Ok((simplex[best_idx].clone(), f_vals[best_idx], n_evals));
        }

        // Compute centroid (excluding worst point)
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
        let f_reflect = f(&x_reflect);
        n_evals += 1;

        if f_reflect < f_vals[best_idx] {
            // Expansion
            let x_expand = reflect(&simplex[worst_idx], &centroid, GAMMA);
            let x_expand = project_bounds(&x_expand, bounds);
            let f_expand = f(&x_expand);
            n_evals += 1;

            if f_expand < f_reflect {
                simplex[worst_idx] = x_expand;
                f_vals[worst_idx] = f_expand;
            } else {
                simplex[worst_idx] = x_reflect;
                f_vals[worst_idx] = f_reflect;
            }
        } else if f_reflect < f_vals[second_worst_idx] {
            // Accept reflection
            simplex[worst_idx] = x_reflect;
            f_vals[worst_idx] = f_reflect;
        } else {
            // Contraction
            let x_contract = if f_reflect < f_vals[worst_idx] {
                // Outside contraction
                reflect(&simplex[worst_idx], &centroid, ALPHA * RHO)
            } else {
                // Inside contraction
                reflect(&simplex[worst_idx], &centroid, -RHO)
            };

            let x_contract = project_bounds(&x_contract, bounds);
            let f_contract = f(&x_contract);
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
                        f_vals[i] = f(&simplex[i]);
                        n_evals += 1;
                    }
                }
            }
        }

        if n_evals >= max_iter {
            break;
        }
    }

    // Return best point
    let best_idx = f_vals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx);

    Ok((simplex[best_idx].clone(), f_vals[best_idx], n_evals))
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
    fn test_nelder_mead_quadratic() {
        // Minimize f(x) = (x-2)² + (y-3)²
        // Minimum at (2, 3)
        let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);

        let x0 = vec![0.0, 0.0];
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        let (x_best, f_best, _) = nelder_mead(f, &x0, &bounds, 1000, 1e-8).unwrap();

        assert!((x_best[0] - 2.0).abs() < 1e-4);
        assert!((x_best[1] - 3.0).abs() < 1e-4);
        assert!(f_best < 1e-6);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        // Global minimum at (1, 1)
        let f = |x: &[f64]| {
            let (a, b) = (1.0, 100.0);
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = vec![0.0, 0.0];
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let (x_best, f_best, n_evals) = nelder_mead(f, &x0, &bounds, 2000, 1e-6).unwrap();

        println!("Rosenbrock: x_best = {x_best:?}, f_best = {f_best}, evals = {n_evals}");

        assert!((x_best[0] - 1.0).abs() < 1e-2);
        assert!((x_best[1] - 1.0).abs() < 1e-2);
        assert!(f_best < 1e-3);
    }

    #[test]
    fn test_nelder_mead_with_bounds() {
        // Minimize f(x,y) = x² + y² with bounds [0,10]×[0,10]
        // Optimal unconstrained is (0,0), but starting at (5,5)
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);

        let x0 = vec![5.0, 5.0];
        let bounds = vec![(0.0, 10.0), (0.0, 10.0)];

        let (x_best, f_best, _) = nelder_mead(f, &x0, &bounds, 1000, 1e-8).unwrap();

        assert!((x_best[0] - 0.0).abs() < 1e-4);
        assert!((x_best[1] - 0.0).abs() < 1e-4);
        assert!(f_best < 1e-6);
    }

    #[test]
    fn test_nelder_mead_1d() {
        // 1D test: minimize (x-3)²
        let f = |x: &[f64]| (x[0] - 3.0).powi(2);

        let x0 = vec![0.0];
        let bounds = vec![(-10.0, 10.0)];

        let (x_best, f_best, _) = nelder_mead(f, &x0, &bounds, 500, 1e-10).unwrap();

        assert!((x_best[0] - 3.0).abs() < 1e-6);
        assert!(f_best < 1e-10);
    }

    #[test]
    fn test_nelder_mead_bounds_mismatch() {
        let f = |x: &[f64]| x[0].powi(2);
        let x0 = vec![0.0, 0.0];
        let bounds = vec![(0.0, 1.0)]; // Wrong length

        let result = nelder_mead(f, &x0, &bounds, 100, 1e-8);
        assert!(result.is_err());
    }

    #[test]
    fn test_nelder_mead_constrained_minimum() {
        // f(x,y) = x + y with bounds [1,2]×[1,2]
        // Constrained minimum at (1, 1)
        let f = |x: &[f64]| x[0] + x[1];

        let x0 = vec![1.5, 1.5];
        let bounds = vec![(1.0, 2.0), (1.0, 2.0)];

        let (x_best, f_best, _) = nelder_mead(f, &x0, &bounds, 500, 1e-8).unwrap();

        assert!((x_best[0] - 1.0).abs() < 1e-4);
        assert!((x_best[1] - 1.0).abs() < 1e-4);
        assert!((f_best - 2.0).abs() < 1e-4);
    }
}
