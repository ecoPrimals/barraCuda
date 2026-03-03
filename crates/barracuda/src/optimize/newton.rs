// SPDX-License-Identifier: AGPL-3.0-or-later
//! Newton-Raphson root-finding algorithm
//!
//! Finds roots of f(x) = 0 using Newton's method with optional numerical
//! derivatives. Converges quadratically near simple roots.
//!
//! # Algorithm
//!
//! x_{n+1} = x_n - f(x_n) / f'(x_n)
//!
//! # Applications
//!
//! - Implicit ODE/PDE solvers
//! - Finding eigenvalues (inverse iteration)
//! - Financial calculations (IRR, bond yield)
//!
//! # References
//!
//! - Numerical Recipes, 3rd Edition, Section 9.4
//! - scipy.optimize.newton

use crate::error::{BarracudaError, Result};

/// Newton-Raphson result
#[derive(Debug, Clone)]
pub struct NewtonResult {
    /// Root found
    pub root: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual |f(root)|
    pub residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// Find root of f(x) = 0 using Newton-Raphson with analytical derivative.
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `df` - Derivative of f
/// * `x0` - Initial guess
/// * `tol` - Tolerance for convergence (|f(x)| < tol)
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// `NewtonResult` containing the root and convergence info
///
/// # Example
///
/// ```
/// use barracuda::optimize::newton;
///
/// // Find √2 by solving x² - 2 = 0
/// let f = |x: f64| x * x - 2.0;
/// let df = |x: f64| 2.0 * x;
///
/// let result = newton(f, df, 1.0, 1e-10, 100)?;
/// assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn newton<F, DF>(f: F, df: DF, x0: f64, tol: f64, max_iter: usize) -> Result<NewtonResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;

    for iter in 0..max_iter {
        let fx = f(x);
        let dfx = df(x);

        // Check convergence
        if fx.abs() < tol {
            return Ok(NewtonResult {
                root: x,
                iterations: iter,
                residual: fx.abs(),
                converged: true,
            });
        }

        // Check for zero derivative
        if dfx.abs() < 1e-14 {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Newton-Raphson: derivative is near zero at x = {x}, f'(x) = {dfx}"
                ),
            });
        }

        // Newton step
        x -= fx / dfx;

        // Check for divergence
        if !x.is_finite() {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Newton-Raphson diverged: x became non-finite after {} iterations",
                    iter + 1
                ),
            });
        }
    }

    // Max iterations reached
    let fx = f(x);
    Ok(NewtonResult {
        root: x,
        iterations: max_iter,
        residual: fx.abs(),
        converged: fx.abs() < tol,
    })
}

/// Find root of f(x) = 0 using Newton-Raphson with numerical derivative.
///
/// Uses central difference approximation for the derivative:
/// f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `x0` - Initial guess
/// * `tol` - Tolerance for convergence (|f(x)| < tol)
/// * `max_iter` - Maximum number of iterations
/// * `h` - Step size for numerical derivative (default: 1e-8)
///
/// # Example
///
/// ```
/// use barracuda::optimize::newton_numerical;
///
/// // Find cube root of 27
/// let f = |x: f64| x * x * x - 27.0;
///
/// let result = newton_numerical(f, 3.5, 1e-10, 100, 1e-8)?;
/// assert!((result.root - 3.0).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn newton_numerical<F>(f: F, x0: f64, tol: f64, max_iter: usize, h: f64) -> Result<NewtonResult>
where
    F: Fn(f64) -> f64,
{
    let mut x = x0;

    for iter in 0..max_iter {
        let fx = f(x);

        // Check convergence
        if fx.abs() < tol {
            return Ok(NewtonResult {
                root: x,
                iterations: iter,
                residual: fx.abs(),
                converged: true,
            });
        }

        // Numerical derivative (central difference)
        let dfx = (f(x + h) - f(x - h)) / (2.0 * h);

        // Check for zero derivative
        if dfx.abs() < 1e-14 {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Newton-Raphson: numerical derivative is near zero at x = {x}, f'(x) ≈ {dfx}"
                ),
            });
        }

        // Newton step
        x -= fx / dfx;

        // Check for divergence
        if !x.is_finite() {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Newton-Raphson diverged: x became non-finite after {} iterations",
                    iter + 1
                ),
            });
        }
    }

    // Max iterations reached
    let fx = f(x);
    Ok(NewtonResult {
        root: x,
        iterations: max_iter,
        residual: fx.abs(),
        converged: fx.abs() < tol,
    })
}

/// Secant method - Newton-like without requiring derivatives.
///
/// Uses two points to approximate the derivative:
/// f'(x) ≈ (f(x_n) - f(x_{n-1})) / (x_n - x_{n-1})
///
/// Convergence is superlinear (order ≈ 1.618) but more robust than Newton.
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `x0` - First initial guess
/// * `x1` - Second initial guess (x0 ≠ x1)
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum iterations
///
/// # Example
///
/// ```
/// use barracuda::optimize::secant;
///
/// let f = |x: f64| x * x - 2.0;
/// let result = secant(f, 1.0, 2.0, 1e-10, 100)?;
/// assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn secant<F>(f: F, mut x0: f64, mut x1: f64, tol: f64, max_iter: usize) -> Result<NewtonResult>
where
    F: Fn(f64) -> f64,
{
    let mut f0 = f(x0);
    let mut f1 = f(x1);

    for iter in 0..max_iter {
        // Check convergence
        if f1.abs() < tol {
            return Ok(NewtonResult {
                root: x1,
                iterations: iter,
                residual: f1.abs(),
                converged: true,
            });
        }

        // Check for identical function values
        if (f1 - f0).abs() < 1e-14 {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Secant method: f(x0) ≈ f(x1), cannot compute approximation. \
                    x0 = {x0}, x1 = {x1}, f0 = {f0}, f1 = {f1}"
                ),
            });
        }

        // Secant step
        let x2 = x1 - f1 * (x1 - x0) / (f1 - f0);

        // Shift
        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);

        // Check for divergence
        if !x1.is_finite() || !f1.is_finite() {
            return Err(BarracudaError::Numerical {
                message: format!(
                    "Secant method diverged: x or f(x) became non-finite after {} iterations",
                    iter + 1
                ),
            });
        }
    }

    Ok(NewtonResult {
        root: x1,
        iterations: max_iter,
        residual: f1.abs(),
        converged: f1.abs() < tol,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    #[test]
    fn test_newton_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;

        let result = newton(f, df, 1.0, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - SQRT_2).abs() < 1e-10);
        assert!(result.iterations < 10); // Newton converges quickly
    }

    #[test]
    fn test_newton_cubic() {
        // x³ - 2x - 5 = 0, root ≈ 2.0946
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let df = |x: f64| 3.0 * x.powi(2) - 2.0;

        let result = newton(f, df, 2.0, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert!(f(result.root).abs() < 1e-10);
    }

    #[test]
    fn test_newton_numerical_sqrt2() {
        let f = |x: f64| x * x - 2.0;

        let result = newton_numerical(f, 1.0, 1e-10, 100, 1e-8).unwrap();
        assert!(result.converged);
        assert!((result.root - SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_newton_numerical_cube_root() {
        let f = |x: f64| x * x * x - 27.0;

        let result = newton_numerical(f, 3.5, 1e-10, 100, 1e-8).unwrap();
        assert!(result.converged);
        assert!((result.root - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_secant_sqrt2() {
        let f = |x: f64| x * x - 2.0;

        let result = secant(f, 1.0, 2.0, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_secant_sin_pi() {
        let f = |x: f64| x.sin();

        let result = secant(f, 3.0, 3.5, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - PI).abs() < 1e-10);
    }

    #[test]
    fn test_newton_zero_derivative() {
        // f(x) = x³ at x = 0 has f'(0) = 0
        let f = |x: f64| x * x * x;
        let df = |x: f64| 3.0 * x * x;

        let result = newton(f, df, 0.0, 1e-10, 100);
        // Should either converge to 0 (the root) or error due to zero derivative
        if let Err(e) = result {
            assert!(e.to_string().contains("zero"));
        }
    }

    #[test]
    fn test_newton_result_fields() {
        let f = |x: f64| x - 1.0;
        let df = |_: f64| 1.0;

        let result = newton(f, df, 0.0, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert!((result.root - 1.0).abs() < 1e-10);
        assert!(result.residual < 1e-10);
        assert!(result.iterations < 100);
    }
}
