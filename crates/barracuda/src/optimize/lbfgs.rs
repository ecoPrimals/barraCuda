// SPDX-License-Identifier: AGPL-3.0-or-later

//! L-BFGS (Limited-memory BFGS) Optimizer
//!
//! The L-BFGS algorithm approximates the BFGS quasi-Newton method using
//! only the most recent `m` gradient/position differences (typically m=5-20),
//! reducing memory from O(n²) to O(mn).
//!
//! This makes L-BFGS suitable for large-scale optimization where full BFGS
//! is prohibitive (thousands to millions of parameters).
//!
//! # Algorithm
//!
//! Uses the two-loop recursion (Nocedal 1980) to compute H⁻¹·g without
//! forming the full Hessian approximation:
//!
//! 1. Backward loop: compute `α_i` = `ρ_i` · `s_i^T` · q for i = k-1, ..., k-m
//! 2. Scale by H₀ = (s^T·y)/(y^T·y) · I
//! 3. Forward loop: compute `β_i` = `ρ_i` · `y_i^T` · r for i = k-m, ..., k-1
//!
//! # References
//! - Nocedal, J. (1980) "Updating quasi-Newton matrices with limited storage"
//! - Liu, D.C. & Nocedal, J. (1989) "On the limited memory BFGS method"
//!
//! Provenance: neuralSpring V70 request → toadStool absorption

use crate::error::{BarracudaError, Result};

/// Configuration for the L-BFGS optimizer.
#[derive(Debug, Clone)]
pub struct LbfgsConfig {
    /// Number of stored correction pairs (memory depth, typically 3-20).
    pub memory: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Gradient norm tolerance for convergence.
    pub gtol: f64,
    /// Function value tolerance for convergence.
    pub ftol: f64,
    /// Line search parameters: sufficient decrease (Wolfe c1).
    pub c1: f64,
    /// Line search parameters: curvature condition (Wolfe c2).
    pub c2: f64,
    /// Maximum line search iterations per step.
    pub max_linesearch: usize,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            memory: 10,
            max_iter: 1000,
            gtol: 1e-6,
            ftol: 1e-8,
            c1: 1e-4,
            c2: 0.9,
            max_linesearch: 20,
        }
    }
}

/// Result of L-BFGS optimization.
#[derive(Debug, Clone)]
pub struct LbfgsResult {
    /// Optimal point found.
    pub x: Vec<f64>,
    /// Function value at the optimal point.
    pub f_val: f64,
    /// Gradient at the optimal point.
    pub gradient: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of function evaluations.
    pub function_evaluations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// L-BFGS optimizer (CPU reference, f64).
///
/// For GPU-accelerated batch optimization of many independent problems,
/// see `BatchedNelderMeadGpu` (gradient-free) or `BfgsGpu` (full Hessian).
///
/// L-BFGS is preferred when:
/// - The problem has thousands+ of parameters
/// - Analytical or cheap numerical gradients are available
/// - Memory is constrained (O(mn) vs O(n²) for full BFGS)
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn lbfgs<F, G>(f: F, grad: G, x0: &[f64], config: &LbfgsConfig) -> Result<LbfgsResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64], &mut [f64]),
{
    let n = x0.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "L-BFGS requires at least 1 dimension".to_string(),
        });
    }
    if config.memory == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "L-BFGS memory must be >= 1".to_string(),
        });
    }

    let m = config.memory;
    let mut x = x0.to_vec();
    let mut g = vec![0.0; n];
    let mut f_val = f(&x);
    grad(&x, &mut g);
    let mut n_evals = 1usize;

    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_history: Vec<f64> = Vec::with_capacity(m);

    let mut x_prev = vec![0.0; n];
    let mut g_prev = vec![0.0; n];

    for iter in 0..config.max_iter {
        let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < config.gtol {
            return Ok(LbfgsResult {
                x,
                f_val,
                gradient: g,
                iterations: iter,
                function_evaluations: n_evals,
                converged: true,
            });
        }

        let d = two_loop_recursion(n, &g, &s_history, &y_history, &rho_history);

        let (alpha, f_new, evals) =
            backtracking_line_search(&f, &x, &d, f_val, &g, config.c1, config.max_linesearch);
        n_evals += evals;

        if (f_val - f_new).abs() < config.ftol * (1.0 + f_val.abs()) && iter > 0 {
            return Ok(LbfgsResult {
                x,
                f_val,
                gradient: g,
                iterations: iter,
                function_evaluations: n_evals,
                converged: true,
            });
        }

        x_prev.copy_from_slice(&x);
        g_prev.copy_from_slice(&g);

        for i in 0..n {
            x[i] += alpha * d[i];
        }
        f_val = f_new;
        grad(&x, &mut g);
        n_evals += 1;

        let mut s_k = vec![0.0; n];
        let mut y_k = vec![0.0; n];
        let mut sy = 0.0;
        for i in 0..n {
            s_k[i] = x[i] - x_prev[i];
            y_k[i] = g[i] - g_prev[i];
            sy += s_k[i] * y_k[i];
        }

        if sy > 1e-30 {
            let rho_k = 1.0 / sy;
            if s_history.len() == m {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s_k);
            y_history.push(y_k);
            rho_history.push(rho_k);
        }
    }

    Ok(LbfgsResult {
        x,
        f_val,
        gradient: g,
        iterations: config.max_iter,
        function_evaluations: n_evals,
        converged: false,
    })
}

/// L-BFGS with numerical gradient (central differences).
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn lbfgs_numerical<F>(f: F, x0: &[f64], config: &LbfgsConfig) -> Result<LbfgsResult>
where
    F: Fn(&[f64]) -> f64,
{
    let h = 1e-7;
    let n = x0.len();
    lbfgs(
        &f,
        |x: &[f64], grad: &mut [f64]| {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            for i in 0..n {
                let orig = x[i];
                x_plus[i] = orig + h;
                x_minus[i] = orig - h;
                grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
                x_plus[i] = orig;
                x_minus[i] = orig;
            }
        },
        x0,
        config,
    )
}

fn two_loop_recursion(
    n: usize,
    g: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    rho_history: &[f64],
) -> Vec<f64> {
    let k = s_history.len();
    let mut q = g.to_vec();
    let mut alpha_buf = vec![0.0; k];

    // Backward loop
    for i in (0..k).rev() {
        let dot: f64 = s_history[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        alpha_buf[i] = rho_history[i] * dot;
        for j in 0..n {
            q[j] -= alpha_buf[i] * y_history[i][j];
        }
    }

    // Scale by initial Hessian approximation: H₀ = γ·I where γ = s^T·y / y^T·y
    if k > 0 {
        let last = k - 1;
        let sy: f64 = s_history[last]
            .iter()
            .zip(y_history[last].iter())
            .map(|(a, b)| a * b)
            .sum();
        let yy: f64 = y_history[last].iter().map(|v| v * v).sum();
        if yy > 1e-30 {
            let gamma = sy / yy;
            for v in &mut q {
                *v *= gamma;
            }
        }
    }

    // Forward loop
    for i in 0..k {
        let dot: f64 = y_history[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let beta = rho_history[i] * dot;
        for j in 0..n {
            q[j] += (alpha_buf[i] - beta) * s_history[i][j];
        }
    }

    // Negate for descent direction: d = -H·g
    for v in &mut q {
        *v = -*v;
    }
    q
}

fn backtracking_line_search<F>(
    f: &F,
    x: &[f64],
    d: &[f64],
    f0: f64,
    g: &[f64],
    c1: f64,
    max_iter: usize,
) -> (f64, f64, usize)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let dg: f64 = d.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
    let mut alpha = 1.0;
    let mut x_trial = vec![0.0; n];
    let mut evals = 0;

    for _ in 0..max_iter {
        for i in 0..n {
            x_trial[i] = x[i] + alpha * d[i];
        }
        let f_trial = f(&x_trial);
        evals += 1;

        if f_trial <= (c1 * alpha).mul_add(dg, f0) {
            return (alpha, f_trial, evals);
        }
        alpha *= 0.5;
    }

    for i in 0..n {
        x_trial[i] = x[i] + alpha * d[i];
    }
    let f_trial = f(&x_trial);
    evals += 1;
    (alpha, f_trial, evals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbfgs_rosenbrock() {
        let rosenbrock = |x: &[f64]| {
            let (a, b) = (1.0, 100.0);
            (a - x[0]).mul_add(a - x[0], b * x[0].mul_add(-x[0], x[1]).powi(2))
        };
        let grad_rosenbrock = |x: &[f64], g: &mut [f64]| {
            let (a, b) = (1.0, 100.0);
            g[0] = (-2.0f64).mul_add(
                a - x[0],
                2.0 * b * x[0].mul_add(-x[0], x[1]) * (-2.0 * x[0]),
            );
            g[1] = 2.0 * b * x[0].mul_add(-x[0], x[1]);
        };

        let config = LbfgsConfig {
            memory: 10,
            max_iter: 2000,
            gtol: 1e-8,
            ..Default::default()
        };

        let result = lbfgs(rosenbrock, grad_rosenbrock, &[-1.0, 1.0], &config).unwrap();
        assert!(result.converged, "L-BFGS did not converge");
        assert!(
            (result.x[0] - 1.0).abs() < 1e-4,
            "x[0] = {}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-4,
            "x[1] = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn test_lbfgs_quadratic() {
        let quad = |x: &[f64]| x[0].mul_add(x[0], 4.0 * x[1] * x[1]);
        let grad_quad = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 8.0 * x[1];
        };

        let result = lbfgs(quad, grad_quad, &[5.0, 3.0], &LbfgsConfig::default()).unwrap();
        assert!(result.converged);
        assert!(result.f_val < 1e-8, "f = {}", result.f_val);
    }

    #[test]
    fn test_lbfgs_numerical_gradient() {
        let quad = |x: &[f64]| x[2].mul_add(x[2], x[0].mul_add(x[0], x[1] * x[1]));
        let result = lbfgs_numerical(quad, &[3.0, 4.0, 5.0], &LbfgsConfig::default()).unwrap();
        assert!(result.converged);
        assert!(result.f_val < 1e-6, "f = {}", result.f_val);
    }

    #[test]
    fn test_lbfgs_memory_depth() {
        let quad = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
        let grad = |x: &[f64], g: &mut [f64]| {
            for (gi, xi) in g.iter_mut().zip(x.iter()) {
                *gi = 2.0 * xi;
            }
        };

        let config = LbfgsConfig {
            memory: 3,
            ..Default::default()
        };
        let x0: Vec<f64> = (0..20).map(|i| (i as f64) + 1.0).collect();
        let result = lbfgs(quad, grad, &x0, &config).unwrap();
        assert!(result.converged);
        assert!(result.f_val < 1e-8);
    }
}
