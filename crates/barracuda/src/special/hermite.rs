// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hermite polynomials
//!
//! Implements the physicist's Hermite polynomials Hₙ(x) using the
//! three-term recurrence relation.
//!
//! # Definition
//!
//! Hₙ(x) = (-1)ⁿ eˣ² dⁿ/dxⁿ e^(-x²)
//!
//! Recurrence: Hₙ₊₁(x) = 2x·Hₙ(x) - 2n·Hₙ₋₁(x)
//!
//! # Applications
//!
//! - Quantum harmonic oscillator wavefunctions
//! - Gaussian quadrature (Gauss-Hermite)
//! - Probability theory (Hermite expansion)
//!
//! # References
//!
//! - Abramowitz & Stegun, §22.3
//! - DLMF 18.3: <https://dlmf.nist.gov/18.3>

/// Compute the physicist's Hermite polynomial Hₙ(x).
///
/// Uses the three-term recurrence relation for stability:
/// - H₀(x) = 1
/// - H₁(x) = 2x
/// - Hₙ₊₁(x) = 2x·Hₙ(x) - 2n·Hₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Polynomial order (non-negative)
/// * `x` - Evaluation point
///
/// # Returns
///
/// The value Hₙ(x).
///
/// # Examples
///
/// ```
/// use barracuda::special::hermite;
///
/// // H₀(x) = 1
/// assert!((hermite(0, 2.0) - 1.0).abs() < 1e-14);
///
/// // H₁(x) = 2x
/// assert!((hermite(1, 2.0) - 4.0).abs() < 1e-14);
///
/// // H₂(x) = 4x² - 2
/// assert!((hermite(2, 2.0) - 14.0).abs() < 1e-14);
///
/// // H₃(x) = 8x³ - 12x
/// assert!((hermite(3, 2.0) - 40.0).abs() < 1e-14);
/// ```
pub fn hermite(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return 2.0 * x;
    }

    // Three-term recurrence: H_{n+1} = 2x·H_n - 2n·H_{n-1}
    let mut h_prev = 1.0; // H₀
    let mut h_curr = 2.0 * x; // H₁

    for k in 1..n {
        let h_next = 2.0 * x * h_curr - 2.0 * (k as f64) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }

    h_curr
}

/// Compute Hermite polynomials H₀(x) through Hₙ(x).
///
/// Returns all orders from 0 to n (inclusive), which is efficient
/// when multiple orders are needed at the same point.
///
/// # Arguments
///
/// * `n` - Maximum polynomial order
/// * `x` - Evaluation point
///
/// # Returns
///
/// Vector of length n+1 containing [H₀(x), H₁(x), ..., Hₙ(x)].
///
/// # Examples
///
/// ```
/// use barracuda::special::hermite::hermite_all;
///
/// let h = hermite_all(3, 1.0);
/// assert_eq!(h.len(), 4);
/// assert!((h[0] - 1.0).abs() < 1e-14);  // H₀(1) = 1
/// assert!((h[1] - 2.0).abs() < 1e-14);  // H₁(1) = 2
/// assert!((h[2] - 2.0).abs() < 1e-14);  // H₂(1) = 4-2 = 2
/// assert!((h[3] - (-4.0)).abs() < 1e-14);  // H₃(1) = 8-12 = -4
/// ```
pub fn hermite_all(n: usize, x: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(n + 1);

    if n == 0 {
        result.push(1.0);
        return result;
    }

    result.push(1.0); // H₀
    result.push(2.0 * x); // H₁

    for k in 1..n {
        let h_next = 2.0 * x * result[k] - 2.0 * (k as f64) * result[k - 1];
        result.push(h_next);
    }

    result
}

/// Compute Hermite polynomial for a batch of x values (CPU path).
///
/// # Arguments
///
/// * `n` - Polynomial order
/// * `x` - Slice of evaluation points
///
/// # Returns
///
/// Vector of Hₙ(x) for each x in the input.
pub fn hermite_batch(n: usize, x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| hermite(n, xi)).collect()
}

/// Normalized Hermite function (wavefunction form).
///
/// ψₙ(x) = (2ⁿ n! √π)^(-1/2) · Hₙ(x) · e^(-x²/2)
///
/// These are the eigenfunctions of the quantum harmonic oscillator.
///
/// # Examples
///
/// ```
/// use barracuda::special::hermite::hermite_normalized;
///
/// // ψ₀(0) = π^(-1/4)
/// let psi0 = hermite_normalized(0, 0.0);
/// assert!((psi0 - std::f64::consts::PI.powf(-0.25)).abs() < 1e-14);
/// ```
pub fn hermite_normalized(n: usize, x: f64) -> f64 {
    use crate::special::factorial::factorial;
    use std::f64::consts::PI;

    let norm = (2.0_f64.powi(n as i32) * factorial(n) * PI.sqrt()).sqrt();
    hermite(n, x) * (-x * x / 2.0).exp() / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_h0() {
        // H₀(x) = 1 for all x
        assert!((hermite(0, 0.0) - 1.0).abs() < 1e-14);
        assert!((hermite(0, 1.0) - 1.0).abs() < 1e-14);
        assert!((hermite(0, -5.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_h1() {
        // H₁(x) = 2x
        assert!((hermite(1, 0.0) - 0.0).abs() < 1e-14);
        assert!((hermite(1, 1.0) - 2.0).abs() < 1e-14);
        assert!((hermite(1, -2.0) - (-4.0)).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_h2() {
        // H₂(x) = 4x² - 2
        assert!((hermite(2, 0.0) - (-2.0)).abs() < 1e-14);
        assert!((hermite(2, 1.0) - 2.0).abs() < 1e-14);
        assert!((hermite(2, 2.0) - 14.0).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_h3() {
        // H₃(x) = 8x³ - 12x
        assert!((hermite(3, 0.0) - 0.0).abs() < 1e-14);
        assert!((hermite(3, 1.0) - (-4.0)).abs() < 1e-14);
        assert!((hermite(3, 2.0) - 40.0).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_h4() {
        // H₄(x) = 16x⁴ - 48x² + 12
        assert!((hermite(4, 0.0) - 12.0).abs() < 1e-14);
        assert!((hermite(4, 1.0) - (-20.0)).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_computed_values() {
        // Physicist's Hermite polynomials via recurrence
        // H₅(x) = 32x⁵ - 160x³ + 120x
        // H₅(1.5) = 32*(7.59375) - 160*(3.375) + 180 = 243 - 540 + 180 = -117
        assert!((hermite(5, 1.5) - (-117.0)).abs() < 1e-10);

        // H₆(x) = 64x⁶ - 480x⁴ + 720x² - 120
        // H₆(2) = 64*64 - 480*16 + 720*4 - 120 = 4096 - 7680 + 2880 - 120 = -824
        assert!((hermite(6, 2.0) - (-824.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_all() {
        let h = hermite_all(4, 1.0);
        assert_eq!(h.len(), 5);
        assert!((h[0] - 1.0).abs() < 1e-14); // H₀
        assert!((h[1] - 2.0).abs() < 1e-14); // H₁
        assert!((h[2] - 2.0).abs() < 1e-14); // H₂
        assert!((h[3] - (-4.0)).abs() < 1e-14); // H₃
        assert!((h[4] - (-20.0)).abs() < 1e-14); // H₄
    }

    #[test]
    fn test_hermite_parity() {
        // Hₙ(-x) = (-1)ⁿ Hₙ(x)
        let x = 1.5;
        for n in 0..8 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((hermite(n, -x) - sign * hermite(n, x)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hermite_batch() {
        let x = vec![0.0, 1.0, 2.0];
        let h2 = hermite_batch(2, &x);
        assert_eq!(h2.len(), 3);
        assert!((h2[0] - (-2.0)).abs() < 1e-14);
        assert!((h2[1] - 2.0).abs() < 1e-14);
        assert!((h2[2] - 14.0).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_normalized_orthonormal() {
        // Normalized Hermite functions should have norm 1
        // ∫ ψₙ²(x) dx = 1
        // We can check that ψ₀(0) = π^(-1/4)
        use std::f64::consts::PI;
        let psi0_at_0 = hermite_normalized(0, 0.0);
        assert!((psi0_at_0 - PI.powf(-0.25)).abs() < 1e-14);
    }

    #[test]
    fn test_hermite_large_order() {
        // Test numerical stability for larger orders
        // H₁₀(1) via recurrence: 8224
        // Verify by checking H₁₀(x) coefficients: (-30240 + 302400x² - 403200x⁴ + 161280x⁶ - 23040x⁸ + 1024x¹⁰)
        // H₁₀(1) = -30240 + 302400 - 403200 + 161280 - 23040 + 1024 = 8224
        let h10 = hermite(10, 1.0);
        assert!((h10 - 8224.0).abs() < 1e-8);
    }
}
