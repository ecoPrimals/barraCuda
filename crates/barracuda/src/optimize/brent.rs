// SPDX-License-Identifier: AGPL-3.0-only
//! Brent's root-finding algorithm
//!
//! Combines the reliability of bisection with the speed of inverse quadratic
//! interpolation and the secant method. Guaranteed to converge like bisection
//! but typically faster.
//!
//! # Algorithm
//!
//! 1. Start with bracketing interval [a, b] where f(a)·f(b) < 0
//! 2. Try inverse quadratic interpolation (if conditions met)
//! 3. Fall back to secant method
//! 4. Fall back to bisection if others would step outside bracket
//!
//! # Applications
//!
//! - General-purpose root-finding
//! - Implicit function evaluation
//! - Eigenvalue problems
//! - Optimization (finding f'(x) = 0)
//!
//! # References
//!
//! - Brent, R. P. (1973). "Algorithms for Minimization without Derivatives"
//! - Numerical Recipes, 3rd Edition, Section 9.3
//! - scipy.optimize.brentq

use crate::error::{BarracudaError, Result};

/// Brent's method result
#[derive(Debug, Clone)]
pub struct BrentResult {
    /// Root found
    pub root: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Final residual |f(root)|
    pub residual: f64,
    /// Function evaluations
    pub function_evaluations: usize,
}

/// Find root of f(x) = 0 using Brent's method.
///
/// Requires f(a) and f(b) to have opposite signs (bracketing).
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `a` - Lower bound of search interval
/// * `b` - Upper bound of search interval
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// `BrentResult` containing the root and convergence info
///
/// # Example
///
/// ```
/// use barracuda::optimize::brent;
///
/// // Find √2 by solving x² - 2 = 0
/// let f = |x: f64| x * x - 2.0;
/// let result = brent(f, 0.0, 2.0, 1e-10, 100)?;
///
/// assert!((result.root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn brent<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<BrentResult>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);
    let mut n_evals = 2;

    // Check bracketing
    if fa * fb > 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Brent's method requires f(a) and f(b) to have opposite signs: \
                f({a}) = {fa}, f({b}) = {fb}"
            ),
        });
    }

    // Ensure |f(a)| >= |f(b)| (so b is the better approximation)
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a; // Previous iterate
    let mut fc = fa;
    let mut d = b - a; // Step before last
    let mut e = d; // Step before d

    for iter in 0..max_iter {
        // Check convergence
        if fb.abs() < tol {
            return Ok(BrentResult {
                root: b,
                iterations: iter,
                residual: fb.abs(),
                function_evaluations: n_evals,
            });
        }

        // Check if interval is small enough
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Ok(BrentResult {
                root: b,
                iterations: iter,
                residual: fb.abs(),
                function_evaluations: n_evals,
            });
        }

        // Decide which method to use
        let mut use_bisection = true;

        if e.abs() >= tol && fa.abs() > fb.abs() {
            // Try interpolation
            let s = fb / fa;

            let (p, q) = if (a - c).abs() < 1e-14 {
                // Secant method (2 distinct points)
                (2.0 * m * s, 1.0 - s)
            } else {
                // Inverse quadratic interpolation (3 distinct points)
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };

            // Adjust signs
            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };

            // Accept interpolation if it stays well within bounds
            if 2.0 * p < 3.0 * m * q - (tol * q).abs() && p < (0.5 * e * q).abs() {
                e = d;
                d = p / q;
                use_bisection = false;
            }
        }

        if use_bisection {
            d = m;
            e = m;
        }

        // Update a to hold the previous best approximation
        a = b;
        fa = fb;

        // Take the step
        if d.abs() > tol {
            b += d;
        } else {
            // Take a minimal step in the direction of the root
            b += if m > 0.0 { tol } else { -tol };
        }

        fb = f(b);
        n_evals += 1;

        // Maintain the bracket
        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            e = b - a;
            d = e;
        }

        // Ensure |f(b)| <= |f(c)|
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
    }

    // Max iterations reached
    Ok(BrentResult {
        root: b,
        iterations: max_iter,
        residual: fb.abs(),
        function_evaluations: n_evals,
    })
}

/// Find a local minimum of f(x) using Brent's method.
///
/// This is Brent's method for optimization (not root-finding).
/// Requires a bracketing triple [a, b, c] where f(b) < f(a) and f(b) < f(c).
///
/// # Arguments
///
/// * `f` - Function to minimize
/// * `a` - Left bound
/// * `b` - Interior point where f(b) < f(a) and f(b) < f(c)
/// * `c` - Right bound
/// * `tol` - Tolerance for convergence
/// * `max_iter` - Maximum iterations
///
/// # Returns
///
/// (`x_min`, `f_min`, iterations)
///
/// # Example
///
/// ```
/// use barracuda::optimize::brent_minimize;
///
/// // Minimize (x - 2)²
/// let f = |x: f64| (x - 2.0).powi(2);
/// let (x_min, f_min, _) = brent_minimize(f, 0.0, 1.5, 4.0, 1e-10, 100)?;
///
/// assert!((x_min - 2.0).abs() < 1e-8);
/// assert!(f_min < 1e-16);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # Errors
///
/// Returns [`Err`] if the minimum cannot be bracketed or convergence fails.
pub fn brent_minimize<F>(
    f: F,
    mut a: f64,
    b: f64,
    mut c: f64,
    tol: f64,
    max_iter: usize,
) -> Result<(f64, f64, usize)>
where
    F: Fn(f64) -> f64,
{
    const GOLDEN: f64 = 0.381_966_011_250_105; // (3 - √5) / 2

    // Ensure a < c
    if a > c {
        std::mem::swap(&mut a, &mut c);
    }

    let mut x = b; // Best point so far
    let mut w = b; // Second best
    let mut v = b; // Previous w
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;

    let mut d: f64 = 0.0; // Step before last
    let mut e: f64 = 0.0; // Step before d

    for iter in 0..max_iter {
        let xm = 0.5 * (a + c);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        // Check convergence
        if (x - xm).abs() <= tol2 - 0.5 * (c - a) {
            return Ok((x, fx, iter));
        }

        let mut use_golden = true;

        // Try parabolic interpolation
        if e.abs() > tol1 {
            // Fit parabola through x, w, v
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            let r = e;
            e = d;

            // Accept parabolic step if it's in bounds and decreasing
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (c - x) {
                d = p / q;
                let u = x + d;

                // Don't evaluate too close to endpoints
                if (u - a) < tol2 || (c - u) < tol2 {
                    d = if x < xm { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            // Golden section step
            e = if x < xm { c - x } else { a - x };
            d = GOLDEN * e;
        }

        // Take the step
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u);

        // Update best points
        if fu <= fx {
            if u < x {
                c = x;
            } else {
                a = x;
            }

            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                c = u;
            }

            if fu <= fw || (w - x).abs() < 1e-14 {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < 1e-14 || (v - w).abs() < 1e-14 {
                v = u;
                fv = fu;
            }
        }
    }

    Ok((x, fx, max_iter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    #[test]
    fn test_brent_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = brent(f, 0.0, 2.0, 1e-10, 100).unwrap();

        assert!((result.root - SQRT_2).abs() < 1e-10);
        assert!(result.residual < 1e-10);
    }

    #[test]
    fn test_brent_sin_pi() {
        let f = |x: f64| x.sin();
        let result = brent(f, 3.0, 4.0, 1e-10, 100).unwrap();

        assert!((result.root - PI).abs() < 1e-10);
    }

    #[test]
    fn test_brent_exp() {
        // Find ln(2)
        let f = |x: f64| x.exp() - 2.0;
        let result = brent(f, 0.0, 1.0, 1e-10, 100).unwrap();

        assert!((result.root - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_brent_cubic() {
        // x³ - 2x - 5 = 0, root between 2 and 3
        let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
        let result = brent(f, 2.0, 3.0, 1e-10, 100).unwrap();

        assert!(f(result.root).abs() < 1e-10);
    }

    #[test]
    fn test_brent_same_sign_error() {
        let f = |x: f64| x - 5.0;
        let result = brent(f, 0.0, 1.0, 1e-10, 100);

        assert!(result.is_err());
    }

    #[test]
    fn test_brent_faster_than_bisection() {
        let f = |x: f64| x * x - 2.0;

        // Brent should converge faster than 50 iterations
        let result = brent(f, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!(
            result.iterations < 15,
            "Brent took {} iterations",
            result.iterations
        );
    }

    #[test]
    fn test_brent_minimize_quadratic() {
        // Minimize (x - 2)²
        let f = |x: f64| (x - 2.0).powi(2);
        let (x_min, f_min, _) = brent_minimize(f, 0.0, 1.5, 4.0, 1e-10, 100).unwrap();

        assert!((x_min - 2.0).abs() < 1e-8);
        assert!(f_min < 1e-14);
    }

    #[test]
    fn test_brent_minimize_cos() {
        // Minimize cos(x) around x = π
        let f = |x: f64| x.cos();
        let (x_min, f_min, _) = brent_minimize(f, 2.0, 3.0, 4.0, 1e-10, 100).unwrap();

        assert!((x_min - PI).abs() < 1e-8);
        assert!((f_min - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_brent_result_fields() {
        let f = |x: f64| x - 1.0;
        let result = brent(f, 0.0, 2.0, 1e-10, 100).unwrap();

        assert!((result.root - 1.0).abs() < 1e-10);
        assert!(result.residual < 1e-10);
        assert!(result.function_evaluations >= 2);
    }
}
