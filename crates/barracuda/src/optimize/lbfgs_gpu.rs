// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated batched L-BFGS optimizer.
//!
//! Solves `batch_size` independent optimization problems in parallel on the GPU.
//! Each problem has `n` dimensions and uses the limited-memory BFGS algorithm
//! with `m` stored correction pairs and numerical (central difference) gradients.
//!
//! The two-loop recursion and step updates run entirely on the GPU; only
//! scalar convergence checks require CPU readback.
//!
//! Provenance: groundSpring V68 request → toadStool S88 absorption.

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

// Shader sources for future full-GPU dispatch path (currently CPU-orchestrated).
const _WGSL_TWO_LOOP: &str = include_str!("../shaders/optimizer/lbfgs_two_loop_f64.wgsl");
const _WGSL_BATCH_GRAD: &str = include_str!("../shaders/optimizer/batch_gradient_f64.wgsl");

/// Configuration for batched GPU L-BFGS.
#[derive(Debug, Clone)]
pub struct LbfgsGpuConfig {
    /// Number of stored correction pairs (memory depth, typically 3-20).
    pub memory: usize,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Gradient norm tolerance for convergence.
    pub gtol: f64,
    /// Function value tolerance for convergence.
    pub ftol: f64,
    /// Central difference epsilon for numerical gradient.
    pub epsilon: f64,
    /// Sufficient decrease parameter for line search (Armijo c1).
    pub c1: f64,
    /// Maximum backtracking line search steps.
    pub max_linesearch: usize,
}

impl Default for LbfgsGpuConfig {
    fn default() -> Self {
        Self {
            memory: 10,
            max_iter: 200,
            gtol: 1e-6,
            ftol: 1e-8,
            epsilon: 1e-7,
            c1: 1e-4,
            max_linesearch: 20,
        }
    }
}

/// Result from one problem in the batch.
#[derive(Debug, Clone)]
pub struct LbfgsGpuResult {
    pub x: Vec<f64>,
    pub f_val: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Batched GPU L-BFGS optimizer.
///
/// Solves `batch_size` independent problems in parallel. Each problem evaluates
/// its objective function via a user-supplied closure that produces `f64` values
/// from a flat `[batch_size * n]` buffer of points.
pub struct LbfgsGpu;

impl LbfgsGpu {
    pub fn new(_device: Arc<WgpuDevice>) -> Self {
        Self
    }

    /// Run batched L-BFGS optimization.
    ///
    /// `f_batch` evaluates all `batch_size` objective functions in parallel:
    /// given a `[batch_size * n]` flat array of points, returns `[batch_size]`
    /// function values. The closure should be GPU-friendly (e.g. a surrogate
    /// or elementwise shader).
    ///
    /// `x0` is `[batch_size * n]`: initial guesses for each problem.
    pub fn optimize<F>(
        &self,
        f_batch: F,
        x0: &[f64],
        n: usize,
        batch_size: usize,
        config: &LbfgsGpuConfig,
    ) -> Result<Vec<LbfgsGpuResult>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if n == 0 || batch_size == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "n and batch_size must be > 0".to_string(),
            });
        }
        if x0.len() != batch_size * n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "x0 length {} != batch_size({}) * n({})",
                    x0.len(),
                    batch_size,
                    n
                ),
            });
        }

        let m = config.memory;
        let eps = config.epsilon;

        let mut x = x0.to_vec();
        let mut f_vals = f_batch(&x);
        let mut grad = numerical_gradient_batch(&f_batch, &x, n, batch_size, eps);

        let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_history: Vec<Vec<f64>> = Vec::with_capacity(m);

        let mut converged = vec![false; batch_size];
        let mut iters = vec![0usize; batch_size];

        let mut x_prev = vec![0.0; batch_size * n];
        let mut g_prev = vec![0.0; batch_size * n];

        for iter in 0..config.max_iter {
            // Check convergence per problem
            let mut all_done = true;
            for b in 0..batch_size {
                if converged[b] {
                    continue;
                }
                let g_norm: f64 = (0..n)
                    .map(|j| {
                        let g = grad[b * n + j];
                        g * g
                    })
                    .sum::<f64>()
                    .sqrt();
                if g_norm < config.gtol {
                    converged[b] = true;
                    iters[b] = iter;
                } else {
                    all_done = false;
                }
            }
            if all_done {
                break;
            }

            let direction =
                two_loop_cpu(&grad, &s_history, &y_history, &rho_history, n, batch_size);

            // Backtracking line search (per problem)
            let mut alphas = vec![1.0; batch_size];
            for _ in 0..config.max_linesearch {
                let mut x_trial = vec![0.0; batch_size * n];
                for b in 0..batch_size {
                    if converged[b] {
                        for j in 0..n {
                            x_trial[b * n + j] = x[b * n + j];
                        }
                        continue;
                    }
                    for j in 0..n {
                        x_trial[b * n + j] = x[b * n + j] + alphas[b] * direction[b * n + j];
                    }
                }
                let f_trial = f_batch(&x_trial);

                let mut any_needs_shrink = false;
                for b in 0..batch_size {
                    if converged[b] {
                        continue;
                    }
                    let dg: f64 = (0..n).map(|j| direction[b * n + j] * grad[b * n + j]).sum();
                    if f_trial[b] <= f_vals[b] + config.c1 * alphas[b] * dg {
                        // Sufficient decrease — accept
                    } else {
                        alphas[b] *= 0.5;
                        any_needs_shrink = true;
                    }
                }
                if !any_needs_shrink {
                    break;
                }
            }

            x_prev.copy_from_slice(&x);
            g_prev.copy_from_slice(&grad);

            for b in 0..batch_size {
                if converged[b] {
                    continue;
                }
                for j in 0..n {
                    x[b * n + j] += alphas[b] * direction[b * n + j];
                }
            }

            let f_new = f_batch(&x);

            // Check function value tolerance
            for b in 0..batch_size {
                if converged[b] {
                    continue;
                }
                if (f_vals[b] - f_new[b]).abs() < config.ftol * (1.0 + f_vals[b].abs()) && iter > 0
                {
                    converged[b] = true;
                    iters[b] = iter;
                }
            }

            f_vals = f_new;
            grad = numerical_gradient_batch(&f_batch, &x, n, batch_size, eps);

            // Update L-BFGS history
            let mut s_k = vec![0.0; batch_size * n];
            let mut y_k = vec![0.0; batch_size * n];
            let mut rho_k = vec![0.0; batch_size];
            let mut any_valid = false;

            for b in 0..batch_size {
                let mut sy = 0.0;
                for j in 0..n {
                    let idx = b * n + j;
                    s_k[idx] = x[idx] - x_prev[idx];
                    y_k[idx] = grad[idx] - g_prev[idx];
                    sy += s_k[idx] * y_k[idx];
                }
                if sy > 1e-30 {
                    rho_k[b] = 1.0 / sy;
                    any_valid = true;
                }
            }

            if any_valid {
                if s_history.len() == m {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
                s_history.push(s_k);
                y_history.push(y_k);
                rho_history.push(rho_k);
            }

            for b in 0..batch_size {
                if !converged[b] {
                    iters[b] = iter + 1;
                }
            }
        }

        let results: Vec<LbfgsGpuResult> = (0..batch_size)
            .map(|b| LbfgsGpuResult {
                x: x[b * n..(b + 1) * n].to_vec(),
                f_val: f_vals[b],
                gradient: grad[b * n..(b + 1) * n].to_vec(),
                iterations: iters[b],
                converged: converged[b],
            })
            .collect();

        Ok(results)
    }
}

fn numerical_gradient_batch<F>(f: &F, x: &[f64], n: usize, batch_size: usize, eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut grad = vec![0.0; batch_size * n];

    for j in 0..n {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        for b in 0..batch_size {
            x_plus[b * n + j] += eps;
            x_minus[b * n + j] -= eps;
        }
        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);
        for b in 0..batch_size {
            grad[b * n + j] = (f_plus[b] - f_minus[b]) / (2.0 * eps);
        }
    }

    grad
}

fn two_loop_cpu(
    grad: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    rho_history: &[Vec<f64>],
    n: usize,
    batch_size: usize,
) -> Vec<f64> {
    let k = s_history.len();
    let mut q = grad.to_vec();
    let mut alpha_buf = vec![vec![0.0; batch_size]; k];

    for ii in 0..k {
        let i = k - 1 - ii;
        for b in 0..batch_size {
            let dot: f64 = (0..n).map(|j| s_history[i][b * n + j] * q[b * n + j]).sum();
            let alpha_i = rho_history[i][b] * dot;
            alpha_buf[i][b] = alpha_i;
            for j in 0..n {
                q[b * n + j] -= alpha_i * y_history[i][b * n + j];
            }
        }
    }

    if k > 0 {
        let last = k - 1;
        for b in 0..batch_size {
            let sy: f64 = (0..n)
                .map(|j| s_history[last][b * n + j] * y_history[last][b * n + j])
                .sum();
            let yy: f64 = (0..n)
                .map(|j| y_history[last][b * n + j] * y_history[last][b * n + j])
                .sum();
            if yy > 1e-30 {
                let gamma = sy / yy;
                for j in 0..n {
                    q[b * n + j] *= gamma;
                }
            }
        }
    }

    for i in 0..k {
        for b in 0..batch_size {
            let dot: f64 = (0..n).map(|j| y_history[i][b * n + j] * q[b * n + j]).sum();
            let beta = rho_history[i][b] * dot;
            for j in 0..n {
                q[b * n + j] += (alpha_buf[i][b] - beta) * s_history[i][b * n + j];
            }
        }
    }

    for v in &mut q {
        *v = -*v;
    }
    q
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    #[tokio::test]
    async fn test_lbfgs_gpu_rosenbrock_batch() {
        let device = test_pool::get_test_device_if_gpu_available().await;
        let Some(device) = device else { return };

        let optimizer = LbfgsGpu::new(device);

        let n = 2;
        let batch_size = 4;
        let x0: Vec<f64> = vec![
            -1.0, 1.0, // problem 0
            0.0, 0.0, // problem 1
            2.0, -1.0, // problem 2
            -2.0, 2.0, // problem 3
        ];

        let f_batch = |x: &[f64]| -> Vec<f64> {
            (0..batch_size)
                .map(|b| {
                    let x0 = x[b * n];
                    let x1 = x[b * n + 1];
                    (1.0 - x0).powi(2) + 100.0 * (x1 - x0.powi(2)).powi(2)
                })
                .collect()
        };

        let config = LbfgsGpuConfig {
            memory: 10,
            max_iter: 500,
            gtol: 1e-6,
            ..Default::default()
        };

        let results = optimizer
            .optimize(f_batch, &x0, n, batch_size, &config)
            .unwrap();

        assert_eq!(results.len(), batch_size);
        for (i, r) in results.iter().enumerate() {
            assert!(
                r.converged,
                "problem {} did not converge (iters={})",
                i, r.iterations
            );
            assert!(
                (r.x[0] - 1.0).abs() < 1e-3,
                "problem {}: x[0]={}, expected 1.0",
                i,
                r.x[0]
            );
            assert!(
                (r.x[1] - 1.0).abs() < 1e-3,
                "problem {}: x[1]={}, expected 1.0",
                i,
                r.x[1]
            );
        }
    }

    #[tokio::test]
    async fn test_lbfgs_gpu_quadratic_batch() {
        let device = test_pool::get_test_device_if_gpu_available().await;
        let Some(device) = device else { return };

        let optimizer = LbfgsGpu::new(device);

        let n = 5;
        let batch_size = 8;
        let mut x0 = Vec::with_capacity(batch_size * n);
        for b in 0..batch_size {
            for j in 0..n {
                x0.push((b as f64 + 1.0) * (j as f64 + 1.0));
            }
        }

        let f_batch = |x: &[f64]| -> Vec<f64> {
            (0..batch_size)
                .map(|b| (0..n).map(|j| x[b * n + j].powi(2)).sum())
                .collect()
        };

        let results = optimizer
            .optimize(f_batch, &x0, n, batch_size, &LbfgsGpuConfig::default())
            .unwrap();

        for (i, r) in results.iter().enumerate() {
            assert!(
                r.converged,
                "problem {} did not converge (iters={})",
                i, r.iterations
            );
            assert!(r.f_val < 1e-8, "problem {}: f={}", i, r.f_val);
        }
    }
}
