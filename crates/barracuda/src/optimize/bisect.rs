//! Bisection root-finding algorithm

use crate::error::{BarracudaError, Result};

/// Find root of f(x) = 0 using bisection method
///
/// Requires f(a) and f(b) to have opposite signs.
///
/// # Arguments
///
/// * `f` - Function to find root of
/// * `a` - Lower bound of search interval
/// * `b` - Upper bound of search interval
/// * `tol` - Tolerance for convergence (|b - a| < tol)
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
///
/// Root x where f(x) ≈ 0, or error if:
/// - f(a) and f(b) have the same sign
/// - Max iterations exceeded without convergence
///
/// # Examples
///
/// ```
/// use barracuda::optimize::bisect;
///
/// // Find √2 by solving x² - 2 = 0
/// let f = |x: f64| x * x - 2.0;
/// let root = bisect(f, 0.0, 2.0, 1e-10, 100)?;
///
/// assert!((root - 2.0_f64.sqrt()).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # References
///
/// - Numerical Recipes, 3rd Edition, Section 9.1
/// - scipy.optimize.bisect
pub fn bisect<F>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);

    // Check that f(a) and f(b) have opposite signs
    if fa * fb > 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "f(a) and f(b) must have opposite signs: f({a}) = {fa}, f({b}) = {fb}"
            ),
        });
    }

    for iter in 0..max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);

        // Check convergence
        if (b - a).abs() < tol || fc.abs() < tol {
            return Ok(c);
        }

        // Narrow the interval
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }

        let _ = fb; // Suppress unused warning

        // Last iteration check
        if iter == max_iter - 1 {
            return Err(BarracudaError::ExecutionError {
                message: format!(
                    "Bisection failed to converge in {max_iter} iterations, interval [{a}, {b}]"
                ),
            });
        }
    }

    Ok((a + b) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_bisect_sqrt2() {
        // Find √2 by solving x² - 2 = 0
        let f = |x: f64| x * x - 2.0;
        let root = bisect(f, 0.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_cubic() {
        // Find root of x³ - x - 2 = 0
        // Root is between 1 and 2
        let f = |x: f64| x.powi(3) - x - 2.0;
        let root = bisect(f, 1.0, 2.0, 1e-10, 100).unwrap();

        // Verify it's actually a root
        assert!(f(root).abs() < 1e-9);
    }

    #[test]
    fn test_bisect_sin() {
        // Find π by solving sin(x) = 0 in [3, 4]
        let f = |x: f64| x.sin();
        let root = bisect(f, 3.0, 4.0, 1e-10, 100).unwrap();
        assert!((root - PI).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_exp() {
        // Find ln(2) by solving e^x - 2 = 0
        let f = |x: f64| x.exp() - 2.0;
        let root = bisect(f, 0.0, 1.0, 1e-10, 100).unwrap();
        assert!((root - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_same_sign_error() {
        // f(0) = -1, f(0.5) = -0.75 (both negative)
        let f = |x: f64| x - 1.0;
        let result = bisect(f, 0.0, 0.5, 1e-10, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_bisect_max_iter_exceeded() {
        // Require impossible tolerance with few iterations
        let f = |x: f64| x * x - 2.0;
        let result = bisect(f, 0.0, 2.0, 1e-15, 5); // Only 5 iterations
        assert!(result.is_err());
    }

    #[test]
    fn test_bisect_exact_root() {
        // f(1.0) = 0 exactly
        let f = |x: f64| x - 1.0;
        let root = bisect(f, 0.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - 1.0).abs() < 1e-10);
    }
}
