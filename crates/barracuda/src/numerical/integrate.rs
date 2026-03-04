// SPDX-License-Identifier: AGPL-3.0-or-later
//! Numerical integration via trapezoidal rule

/// WGSL kernel for trapezoidal-rule numerical integration (f64).
pub const WGSL_TRAPZ_F64: &str = include_str!("../shaders/math/trapz_f64.wgsl");

use crate::error::{BarracudaError, Result};

/// Trapezoidal integration: ∫ y(x) dx
///
/// Computes the integral using the trapezoidal rule:
/// ∫ y(x) dx ≈ Σᵢ (y[i] + y[i+1])/2 · (x[i+1] - x[i])
///
/// # Arguments
///
/// * `y` - Function values [y(x₀), y(x₁), ..., y(xₙ)]
/// * `x` - Grid points [x₀, x₁, ..., xₙ] (need not be uniform)
///
/// # Returns
///
/// Approximate integral value
///
/// # Errors
///
/// Returns error if x and y have different lengths or if x is not strictly increasing.
///
/// # Examples
///
/// ```
/// use barracuda::numerical::trapz;
///
/// // Integrate y = x from 0 to 4
/// let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
/// let integral = trapz(&y, &x)?;
///
/// // ∫₀⁴ x dx = x²/2 |₀⁴ = 8
/// assert!((integral - 8.0).abs() < 1e-10);
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
///
/// # References
///
/// - numpy.trapz: <https://numpy.org/doc/stable/reference/generated/numpy.trapz.html>
pub fn trapz(y: &[f64], x: &[f64]) -> Result<f64> {
    if y.len() != x.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!("y and x must have same length: {} vs {}", y.len(), x.len()),
        });
    }

    let n = y.len();

    if n == 0 {
        return Ok(0.0);
    }

    if n == 1 {
        return Ok(0.0);
    }

    // Verify x is strictly increasing (or at least non-decreasing)
    for i in 1..n {
        if x[i] < x[i - 1] {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "x must be non-decreasing: x[{}] = {} < x[{}] = {}",
                    i,
                    x[i],
                    i - 1,
                    x[i - 1]
                ),
            });
        }
    }

    let mut integral = 0.0;

    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * (y[i] + y[i + 1]) * dx;
    }

    Ok(integral)
}

/// Weighted trapezoidal product integral: ∫ f(x)·g₁(x)·g₂(x)·w(x) dx
///
/// Used in physics calculations (e.g., HFB matrix elements).
///
/// # Arguments
///
/// * `f` - First function values
/// * `g1` - Second function values
/// * `g2` - Third function values
/// * `x` - Grid points
/// * `weights` - Quadrature weights
///
/// # Returns
///
/// Approximate integral value
///
/// # Examples
///
/// ```
/// use barracuda::numerical::trapz_product;
///
/// let n = 5;
/// let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
/// let f = vec![1.0; n];
/// let g1 = x.clone();
/// let g2 = x.clone();
/// let weights = vec![1.0; n];
///
/// // ∫ 1 · x · x · 1 dx = ∫ x² dx
/// let integral = trapz_product(&f, &g1, &g2, &x, &weights)?;
/// assert!((integral - 21.33).abs() < 1.0);  // ∫₀⁴ x² dx = 64/3 ≈ 21.33 (trapezoidal approximation)
/// # Ok::<(), barracuda::error::BarracudaError>(())
/// ```
pub fn trapz_product(f: &[f64], g1: &[f64], g2: &[f64], x: &[f64], weights: &[f64]) -> Result<f64> {
    if f.len() != g1.len() || f.len() != g2.len() || f.len() != x.len() || f.len() != weights.len()
    {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "All arrays must have same length: f={}, g1={}, g2={}, x={}, weights={}",
                f.len(),
                g1.len(),
                g2.len(),
                x.len(),
                weights.len()
            ),
        });
    }

    let n = f.len();

    if n == 0 {
        return Ok(0.0);
    }

    if n == 1 {
        return Ok(f[0] * g1[0] * g2[0] * weights[0]);
    }

    let mut integral = 0.0;

    for i in 0..n - 1 {
        let product_i = f[i] * g1[i] * g2[i] * weights[i];
        let product_i1 = f[i + 1] * g1[i + 1] * g2[i + 1] * weights[i + 1];
        let dx = x[i + 1] - x[i];
        integral += 0.5 * (product_i + product_i1) * dx;
    }

    Ok(integral)
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapz_linear() {
        // ∫₀⁴ x dx = x²/2 |₀⁴ = 8
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = trapz(&y, &x).unwrap();
        assert!((result - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapz_quadratic() {
        // ∫₀⁴ x² dx = x³/3 |₀⁴ = 64/3 ≈ 21.333
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
        let result = trapz(&y, &x).unwrap();
        let expected = 64.0 / 3.0;
        println!(
            "trapz_quadratic: result = {}, expected = {}, error = {}",
            result,
            expected,
            (result - expected).abs()
        );
        assert!((result - expected).abs() < 0.7); // Trapz has O(h²) error, coarse grid
    }

    #[test]
    fn test_trapz_constant() {
        // ∫₀⁴ 5 dx = 5·4 = 20
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0; 5];
        let result = trapz(&y, &x).unwrap();
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapz_nonuniform_grid() {
        // ∫₀¹⁰ x dx with non-uniform spacing
        let x = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let y = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        let result = trapz(&y, &x).unwrap();
        let expected = 50.0; // x²/2 |₀¹⁰ = 50
        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_trapz_empty() {
        let result = trapz(&[], &[]).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_trapz_single_point() {
        let result = trapz(&[42.0], &[1.0]).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_trapz_mismatched_lengths() {
        let result = trapz(&[1.0, 2.0, 3.0], &[0.0, 1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_trapz_non_increasing() {
        let x = vec![0.0, 2.0, 1.0, 3.0]; // Not increasing
        let y = vec![0.0, 2.0, 1.0, 3.0];
        let result = trapz(&y, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_trapz_product_simple() {
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let f = vec![1.0; n];
        let g1 = x.clone();
        let g2 = x.clone();
        let weights = vec![1.0; n];

        // ∫ 1·x·x·1 dx = ∫ x² dx
        let result = trapz_product(&f, &g1, &g2, &x, &weights).unwrap();
        let expected = 64.0 / 3.0; // ∫₀⁴ x² dx
        println!(
            "trapz_product_simple: result = {}, expected = {}, error = {}",
            result,
            expected,
            (result - expected).abs()
        );
        assert!((result - expected).abs() < 0.7); // Coarse grid, larger error
    }

    #[test]
    fn test_trapz_product_weights() {
        let _n = 3;
        let x = vec![0.0, 1.0, 2.0];
        let f = vec![1.0, 2.0, 3.0];
        let g1 = vec![1.0, 1.0, 1.0];
        let g2 = vec![1.0, 1.0, 1.0];
        let weights = vec![2.0, 2.0, 2.0];

        // ∫ f·1·1·2 dx = ∫ 2f dx
        let result = trapz_product(&f, &g1, &g2, &x, &weights).unwrap();

        // Manual: 0.5 * ((1*1*1*2) + (2*1*1*2)) * 1 + 0.5 * ((2*1*1*2) + (3*1*1*2)) * 1
        //       = 0.5 * (2 + 4) * 1 + 0.5 * (4 + 6) * 1
        //       = 3 + 5 = 8
        assert!((result - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapz_product_mismatched_lengths() {
        let result = trapz_product(
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[0.0, 1.0],
            &[1.0], // Wrong length
        );
        assert!(result.is_err());
    }
}
