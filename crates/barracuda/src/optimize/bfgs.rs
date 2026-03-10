// SPDX-License-Identifier: AGPL-3.0-only
//! BFGS Quasi-Newton Optimizer
//!
//! The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is a quasi-Newton
//! optimization method that builds an approximation to the inverse Hessian
//! matrix using gradient information.
//!
//! # Algorithm
//!
//! BFGS iteratively updates:
//! 1. Search direction: d = -H·∇f
//! 2. Line search: α = argmin f(x + α·d)
//! 3. Position update: x ← x + α·d
//! 4. Hessian approximation update using BFGS formula
//!
//! # Applications
//!
//! - **Parameter fitting**: Physics models, neural networks
//! - **Maximum likelihood**: Statistical inference
//! - **Energy minimization**: Molecular geometry optimization
//! - **Control optimization**: Trajectory planning
//!
//! # References
//!
//! - Nocedal, J. & Wright, S. (2006), "Numerical Optimization"
//! - Broyden, C. G. (1970)

use crate::error::{BarracudaError, Result};

/// Maximum iterations for difficult optimization problems (e.g. Rosenbrock).
#[allow(dead_code, reason = "used in test_bfgs_rosenbrock")]
const BFGS_MAX_ITER_EXTENDED: usize = 5000;

/// GPU shader for BFGS inverse Hessian update (O(n²) parallel).
///
/// f64 canonical — f32 derived via downcast when needed.
const WGSL_BFGS_UPDATE_F64: &str = include_str!("../shaders/optimizer/bfgs_update_f64.wgsl");

/// Entry points: `bfgs_update`, `dot_product`, `mat_vec_mul`, `compute_Hy_and_yHy`.
pub static WGSL_BFGS_UPDATE: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_BFGS_UPDATE_F64.to_string());

/// GPU shader for batch numerical gradient via central/forward differences.
///
/// Entry points: `central_difference`, `forward_difference`, `generate_perturbed_points`.
const WGSL_BATCH_GRADIENT_F64: &str = include_str!("../shaders/optimizer/batch_gradient_f64.wgsl");

/// WGSL batch gradient shader (f32, downcast from f64).
pub static WGSL_BATCH_GRADIENT: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_BATCH_GRADIENT_F64.to_string());

/// Configuration for the BFGS optimizer.
#[derive(Debug, Clone)]
pub struct BfgsConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Gradient norm tolerance for convergence
    pub gtol: f64,
    /// Step size tolerance for convergence
    pub xtol: f64,
    /// Function value tolerance for convergence
    pub ftol: f64,
    /// Line search parameters (c1, c2 for Wolfe conditions)
    pub c1: f64,
    /// Second Wolfe condition parameter (curvature)
    pub c2: f64,
    /// Maximum line search iterations
    pub max_linesearch: usize,
}

impl Default for BfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            gtol: 1e-6,
            xtol: 1e-8,
            ftol: 1e-8,
            c1: 1e-4,
            c2: 0.9,
            max_linesearch: 20,
        }
    }
}

impl BfgsConfig {
    /// Create a new configuration with specified tolerances.
    #[must_use]
    pub fn new(gtol: f64, max_iter: usize) -> Self {
        Self {
            max_iter,
            gtol,
            ..Default::default()
        }
    }
}

/// Result of BFGS optimization.
#[derive(Debug)]
pub struct BfgsResult {
    /// Optimal point found
    pub x: Vec<f64>,
    /// Function value at optimum
    pub f: f64,
    /// Gradient at optimum
    pub grad: Vec<f64>,
    /// Number of iterations
    pub n_iter: usize,
    /// Number of function evaluations
    pub n_feval: usize,
    /// Number of gradient evaluations
    pub n_geval: usize,
    /// Convergence status
    pub converged: bool,
    /// Reason for termination
    pub message: String,
}

/// BFGS optimizer for unconstrained minimization.
///
/// # Arguments
///
/// * `f` - Objective function f(x) -> f64
/// * `grad_f` - Gradient function ∇f(x) -> `Vec<f64>`
/// * `x0` - Initial guess
/// * `config` - Optimizer configuration
///
/// # Returns
///
/// [`BfgsResult`] containing the solution and diagnostics.
///
/// # Example
///
/// ```
/// use barracuda::optimize::bfgs::{bfgs, BfgsConfig};
///
/// // Minimize f(x) = x₀² + x₁²
/// let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
/// let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
///
/// let config = BfgsConfig::default();
/// let result = bfgs(&f, &grad, &[1.0, 1.0], &config).unwrap();
///
/// assert!(result.x[0].abs() < 1e-5);
/// assert!(result.x[1].abs() < 1e-5);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn bfgs<F, G>(f: &F, grad_f: &G, x0: &[f64], config: &BfgsConfig) -> Result<BfgsResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "x0 must not be empty".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut gx = grad_f(&x);
    let mut n_feval = 1;
    let mut n_geval = 1;

    // Initialize inverse Hessian approximation as identity
    let mut h_inv = identity_matrix(n);

    for iter in 0..config.max_iter {
        // Check gradient convergence
        let grad_norm = gx.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < config.gtol {
            return Ok(BfgsResult {
                x,
                f: fx,
                grad: gx,
                n_iter: iter,
                n_feval,
                n_geval,
                converged: true,
                message: "Gradient norm below tolerance".to_string(),
            });
        }

        // Compute search direction: d = -H⁻¹·g
        let d = mat_vec_mul(&h_inv, &gx)
            .into_iter()
            .map(|di| -di)
            .collect::<Vec<_>>();

        // Line search with backtracking
        let (alpha, fx_new, ls_evals) = backtracking_line_search(f, &x, &d, fx, &gx, config)?;
        n_feval += ls_evals;

        // Check for sufficient decrease
        if (fx - fx_new).abs() < config.ftol {
            return Ok(BfgsResult {
                x,
                f: fx,
                grad: gx,
                n_iter: iter,
                n_feval,
                n_geval,
                converged: true,
                message: "Function value change below tolerance".to_string(),
            });
        }

        // Update position: x_new = x + α·d
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();

        // Compute step and gradient difference
        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(xn, xo)| xn - xo).collect();
        let step_norm = s.iter().map(|si| si * si).sum::<f64>().sqrt();

        if step_norm < config.xtol {
            return Ok(BfgsResult {
                x: x_new,
                f: fx_new,
                grad: gx,
                n_iter: iter,
                n_feval,
                n_geval,
                converged: true,
                message: "Step size below tolerance".to_string(),
            });
        }

        // Compute new gradient
        let gx_new = grad_f(&x_new);
        n_geval += 1;

        let y: Vec<f64> = gx_new
            .iter()
            .zip(gx.iter())
            .map(|(gn, go)| gn - go)
            .collect();

        // BFGS update: H⁻¹ ← (I - ρsy^T)H⁻¹(I - ρys^T) + ρss^T
        let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

        // Skip update if curvature condition is not satisfied
        if sy > 1e-10 {
            let rho = 1.0 / sy;
            bfgs_update(&mut h_inv, &s, &y, rho);
        }

        x = x_new;
        fx = fx_new;
        gx = gx_new;
    }

    Ok(BfgsResult {
        x,
        f: fx,
        grad: gx,
        n_iter: config.max_iter,
        n_feval,
        n_geval,
        converged: false,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Backtracking line search satisfying Armijo condition.
fn backtracking_line_search<F>(
    f: &F,
    x: &[f64],
    d: &[f64],
    fx: f64,
    gx: &[f64],
    config: &BfgsConfig,
) -> Result<(f64, f64, usize)>
where
    F: Fn(&[f64]) -> f64,
{
    let mut alpha = 1.0;
    let dg: f64 = d.iter().zip(gx.iter()).map(|(di, gi)| di * gi).sum(); // directional derivative

    if dg >= 0.0 {
        return Err(BarracudaError::Numerical {
            message: "Search direction is not a descent direction".to_string(),
        });
    }

    let mut n_evals = 0;
    let rho = 0.5; // backtracking factor

    for _ in 0..config.max_linesearch {
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let fx_new = f(&x_new);
        n_evals += 1;

        // Armijo condition: f(x + αd) ≤ f(x) + c₁·α·∇f·d
        if fx_new <= fx + config.c1 * alpha * dg {
            return Ok((alpha, fx_new, n_evals));
        }

        alpha *= rho;
    }

    // Return best found even if line search didn't fully succeed
    let x_final: Vec<f64> = x
        .iter()
        .zip(d.iter())
        .map(|(xi, di)| xi + alpha * di)
        .collect();
    Ok((alpha, f(&x_final), n_evals + 1))
}

/// BFGS inverse Hessian update.
fn bfgs_update(h_inv: &mut [Vec<f64>], s: &[f64], y: &[f64], rho: f64) {
    let n = s.len();

    // Hy = H⁻¹·y
    let hy = mat_vec_mul(h_inv, y);

    // yHy = y^T·H⁻¹·y
    let yhy: f64 = y.iter().zip(hy.iter()).map(|(yi, hyi)| yi * hyi).sum();

    // Update H⁻¹ using Sherman-Morrison-Woodbury-like formula
    // H_new = (I - ρsy^T)H(I - ρys^T) + ρss^T
    // = H - ρ(sy^T)H - ρH(ys^T) + ρ²(sy^T)H(ys^T) + ρss^T
    // = H - ρ(s⊗Hy) - ρ(Hy⊗s) + ρ²(yHy)(s⊗s) + ρ(s⊗s)
    // = H - ρ(s⊗Hy + Hy⊗s) + (ρ²yHy + ρ)(s⊗s)
    // = H - ρ(s⊗Hy + Hy⊗s) + ρ(1 + ρyHy)(s⊗s)

    let factor = rho * (1.0 + rho * yhy);

    for i in 0..n {
        for j in 0..n {
            h_inv[i][j] += factor * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
        }
    }
}

/// Create n×n identity matrix.
fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![0.0; n]; n];
    for i in 0..n {
        mat[i][i] = 1.0;
    }
    mat
}

/// Matrix-vector multiplication.
fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Compute numerical gradient using central differences.
///
/// Useful when analytical gradients are not available.
pub fn numerical_gradient<F>(f: &F, x: &[f64], eps: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..n {
        x_plus[i] = x[i] + eps;
        x_minus[i] = x[i] - eps;

        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}

/// BFGS with numerical gradient (convenience wrapper).
///
/// Uses central difference approximation for gradients.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn bfgs_numerical<F>(f: &F, x0: &[f64], config: &BfgsConfig) -> Result<BfgsResult>
where
    F: Fn(&[f64]) -> f64,
{
    let eps = 1e-8;
    let grad_f = |x: &[f64]| numerical_gradient(f, x, eps);
    bfgs(f, &grad_f, x0, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfgs_quadratic() {
        // Minimize f(x) = x₀² + 2x₁²
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let grad = |x: &[f64]| vec![2.0 * x[0], 4.0 * x[1]];

        let config = BfgsConfig::default();
        let result = bfgs(&f, &grad, &[1.0, 1.0], &config).unwrap();

        assert!(result.converged);
        assert!(result.x[0].abs() < 1e-3, "x[0] = {}", result.x[0]);
        assert!(result.x[1].abs() < 1e-3, "x[1] = {}", result.x[1]);
        assert!(result.f < 1e-5, "f = {}", result.f);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        // Rosenbrock function: f(x) = (1-x₀)² + 100(x₁-x₀²)²
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let grad = |x: &[f64]| {
            vec![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
                200.0 * (x[1] - x[0].powi(2)),
            ]
        };

        let config = BfgsConfig {
            max_iter: BFGS_MAX_ITER_EXTENDED,
            gtol: 1e-6,
            ..Default::default()
        };

        let result = bfgs(&f, &grad, &[0.0, 0.0], &config).unwrap();

        assert!(result.converged, "Failed to converge: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-4, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn test_bfgs_numerical_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

        let config = BfgsConfig::default();
        let result = bfgs_numerical(&f, &[1.0, 1.0], &config).unwrap();

        assert!(result.converged);
        assert!(result.x[0].abs() < 1e-4);
        assert!(result.x[1].abs() < 1e-4);
    }

    #[test]
    fn test_numerical_gradient() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1] + x[0] * x[1];
        let x = vec![1.0, 2.0];

        let grad = numerical_gradient(&f, &x, 1e-6);

        // Analytical: ∂f/∂x₀ = 2x₀ + x₁ = 4
        //             ∂f/∂x₁ = 4x₁ + x₀ = 9
        assert!((grad[0] - 4.0).abs() < 1e-4);
        assert!((grad[1] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_bfgs_1d() {
        // f(x) = (x-2)²
        let f = |x: &[f64]| (x[0] - 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];

        let config = BfgsConfig::default();
        let result = bfgs(&f, &grad, &[0.0], &config).unwrap();

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bfgs_high_dim() {
        // f(x) = Σᵢ xᵢ²
        let n = 10;
        let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
        let grad = |x: &[f64]| x.iter().map(|xi| 2.0 * xi).collect();

        let x0: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let config = BfgsConfig::default();
        let result = bfgs(&f, &grad, &x0, &config).unwrap();

        assert!(result.converged);
        for (i, xi) in result.x.iter().enumerate() {
            assert!(xi.abs() < 1e-5, "x[{i}] = {xi}, expected ~0");
        }
    }

    #[test]
    fn test_config_builder() {
        let config = BfgsConfig::new(1e-8, 500);
        assert!((config.gtol - 1e-8).abs() < 1e-14);
        assert_eq!(config.max_iter, 500);
    }
}
