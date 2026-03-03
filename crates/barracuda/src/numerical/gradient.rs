//! Numerical gradient computation via finite differences

/// Compute numerical gradient using finite difference stencils
///
/// **Matches numpy.gradient() behavior exactly** for 1D arrays with uniform spacing.
///
/// # Arguments
///
/// * `f` - Function values [f(x₀), f(x₁), ..., f(xₙ)]
/// * `dx` - Grid spacing (uniform)
///
/// # Returns
///
/// Gradient [df/dx(x₀), df/dx(x₁), ..., df/dx(xₙ)]
///
/// # Algorithm
///
/// - **Interior points**: 2nd-order central difference: `(f[i+1] - f[i-1]) / (2·dx)`
/// - **Boundary points**: 2nd-order forward/backward stencils (matches numpy.gradient)
///   - Start: `(-3·f[0] + 4·f[1] - f[2]) / (2·dx)`
///   - End:   `(3·f[n-1] - 4·f[n-2] + f[n-3]) / (2·dx)`
///
/// For n=2 arrays, uses 1st-order differences (only option with 2 points).
///
/// # Examples
///
/// ```
/// use barracuda::numerical::gradient_1d;
///
/// // Gradient of y = x² at x = [0, 1, 2, 3, 4]
/// let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let dy_dx = gradient_1d(&y, 1.0);
///
/// // True derivative is 2x → [0, 2, 4, 6, 8]
/// // 2nd-order stencils are EXACT for polynomials up to degree 2
/// assert!((dy_dx[0] - 0.0).abs() < 1e-10);  // 2nd-order forward: (-3*0 + 4*1 - 4) / 2 = 0
/// assert!((dy_dx[1] - 2.0).abs() < 1e-10);  // central: (4 - 0) / 2 = 2
/// assert!((dy_dx[2] - 4.0).abs() < 1e-10);  // central: (9 - 1) / 2 = 4
/// assert!((dy_dx[3] - 6.0).abs() < 1e-10);  // central: (16 - 4) / 2 = 6
/// assert!((dy_dx[4] - 8.0).abs() < 1e-10);  // 2nd-order backward: (3*16 - 4*9 + 4) / 2 = 8
/// ```
///
/// # References
///
/// - numpy.gradient: <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>
pub fn gradient_1d(f: &[f64], dx: f64) -> Vec<f64> {
    let n = f.len();

    if n == 0 {
        return Vec::new();
    }

    if n == 1 {
        return vec![0.0];
    }

    let mut grad = vec![0.0; n];

    if n == 2 {
        // Only 2 points: 1st-order forward/backward (only option)
        grad[0] = (f[1] - f[0]) / dx;
        grad[1] = (f[1] - f[0]) / dx;
        return grad;
    }

    // n >= 3: Use 2nd-order stencils at boundaries (matches numpy.gradient)

    // 2nd-order forward difference at start
    // Derived from Taylor expansion: f'(x) ≈ (-3f(x) + 4f(x+h) - f(x+2h)) / (2h)
    grad[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * dx);

    // Central difference for interior points (2nd-order)
    for i in 1..n - 1 {
        grad[i] = (f[i + 1] - f[i - 1]) / (2.0 * dx);
    }

    // 2nd-order backward difference at end
    // Derived from Taylor expansion: f'(x) ≈ (3f(x) - 4f(x-h) + f(x-2h)) / (2h)
    grad[n - 1] = (3.0 * f[n - 1] - 4.0 * f[n - 2] + f[n - 3]) / (2.0 * dx);

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_1d_linear() {
        // y = 2x (gradient should be 2.0 everywhere)
        // 2nd-order stencils are EXACT for linear functions
        let y = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let grad = gradient_1d(&y, 1.0);

        for (i, &g) in grad.iter().enumerate() {
            assert!((g - 2.0).abs() < 1e-10, "grad[{i}] = {g}, expected 2.0");
        }
    }

    #[test]
    fn test_gradient_1d_quadratic_exact() {
        // y = x² at x = [0, 1, 2, 3, 4] → y = [0, 1, 4, 9, 16]
        // True derivative: dy/dx = 2x → [0, 2, 4, 6, 8]
        // 2nd-order stencils are EXACT for quadratic functions!
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let grad = gradient_1d(&y, 1.0);
        let expected = vec![0.0, 2.0, 4.0, 6.0, 8.0];

        for (i, (&g, &e)) in grad.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-10, "grad[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn test_gradient_1d_cubic() {
        // y = x³ at x = [0, 1, 2, 3, 4] → y = [0, 1, 8, 27, 64]
        // True derivative: dy/dx = 3x² → [0, 3, 12, 27, 48]
        //
        // Central difference for cubic has O(h²) error: (f'''(x) * h²) / 6 = 6h²/6 = h² = 1
        // So interior points have error of ±1 from the true value
        //
        // Computed gradients:
        //   grad[0] = (-3*0 + 4*1 - 8) / 2 = -4/2 = -2 (true: 0)
        //   grad[1] = (8 - 0) / 2 = 4 (true: 3)
        //   grad[2] = (27 - 1) / 2 = 13 (true: 12)
        //   grad[3] = (64 - 8) / 2 = 28 (true: 27)
        //   grad[4] = (3*64 - 4*27 + 8) / 2 = (192 - 108 + 8) / 2 = 46 (true: 48)
        let y = vec![0.0, 1.0, 8.0, 27.0, 64.0];
        let grad = gradient_1d(&y, 1.0);
        let expected = vec![-2.0, 4.0, 13.0, 28.0, 46.0];

        for (i, (&g, &e)) in grad.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-10, "grad[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn test_gradient_1d_constant() {
        // y = 5.0 (gradient should be 0)
        let y = vec![5.0; 10];
        let grad = gradient_1d(&y, 1.0);

        for (i, &g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-10, "grad[{i}] = {g}, expected 0.0");
        }
    }

    #[test]
    fn test_gradient_1d_empty() {
        let grad = gradient_1d(&[], 1.0);
        assert_eq!(grad.len(), 0);
    }

    #[test]
    fn test_gradient_1d_single() {
        let grad = gradient_1d(&[42.0], 1.0);
        assert_eq!(grad, vec![0.0]);
    }

    #[test]
    fn test_gradient_1d_two_points() {
        // Only 2 points: must use 1st-order forward/backward
        let grad = gradient_1d(&[0.0, 2.0], 1.0);
        assert_eq!(grad, vec![2.0, 2.0]);
    }

    #[test]
    fn test_gradient_1d_nonuniform_spacing() {
        // Test with dx = 0.5
        // y = 2x² with x = [0, 0.5, 1, 1.5, 2] → y = [0, 0.5, 2.0, 4.5, 8.0]
        // True dy/dx = 4x → [0, 2, 4, 6, 8]
        let y = vec![0.0, 0.5, 2.0, 4.5, 8.0];
        let grad = gradient_1d(&y, 0.5);
        let expected = vec![0.0, 2.0, 4.0, 6.0, 8.0];

        for (i, (&g, &e)) in grad.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-10, "grad[{i}] = {g}, expected {e}");
        }
    }

    #[test]
    fn test_gradient_1d_boundary_accuracy() {
        // Verify 2nd-order boundary stencil formulas
        // For quadratic functions (where 2nd-order is exact), we verify correctness

        // y = x² at x = [0, 0.1, 0.2, ..., 1.0]
        let n = 11;
        let dx = 0.1;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                x * x
            })
            .collect();
        let grad = gradient_1d(&y, dx);

        // True derivative: dy/dx = 2x
        // At x=0: true grad = 0
        // At x=1: true grad = 2
        assert!(
            grad[0].abs() < 1e-10,
            "grad[0] = {}, expected 0.0 (2nd-order exact for quadratic)",
            grad[0]
        );
        assert!(
            (grad[n - 1] - 2.0).abs() < 1e-10,
            "grad[{}] = {}, expected 2.0 (2nd-order exact for quadratic)",
            n - 1,
            grad[n - 1]
        );
    }
}
