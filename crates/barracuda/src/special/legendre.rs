// SPDX-License-Identifier: AGPL-3.0-only
//! Legendre polynomials and associated Legendre functions
//!
//! Implements Pₙ(x) and Pₙᵐ(x) using recurrence relations.
//!
//! # Definitions
//!
//! Legendre polynomial: Pₙ(x) = (1/2ⁿn!) dⁿ/dxⁿ (x²-1)ⁿ
//!
//! Associated Legendre: Pₙᵐ(x) = (-1)ᵐ (1-x²)^(m/2) dᵐ/dxᵐ Pₙ(x)
//!
//! # Applications
//!
//! - Spherical harmonics: Yₗᵐ(θ,φ) involves Pₗᵐ(cos θ)
//! - Multipole expansion
//! - Angular momentum coupling
//! - Gravitational potential of deformed bodies
//!
//! # References
//!
//! - Abramowitz & Stegun, §8.1, §8.6
//! - DLMF 14: <https://dlmf.nist.gov/14>

/// Compute the Legendre polynomial Pₙ(x).
///
/// Uses the three-term recurrence relation for stability:
/// - P₀(x) = 1
/// - P₁(x) = x
/// - (n+1)Pₙ₊₁(x) = (2n+1)x·Pₙ(x) - n·Pₙ₋₁(x)
///
/// # Arguments
///
/// * `n` - Polynomial order (non-negative)
/// * `x` - Evaluation point, typically in [-1, 1]
///
/// # Returns
///
/// The value Pₙ(x).
///
/// # Examples
///
/// ```
/// use barracuda::special::legendre;
///
/// // P₀(x) = 1
/// assert!((legendre(0, 0.5) - 1.0).abs() < 1e-14);
///
/// // P₁(x) = x
/// assert!((legendre(1, 0.5) - 0.5).abs() < 1e-14);
///
/// // P₂(x) = (3x² - 1)/2
/// assert!((legendre(2, 0.5) - (-0.125)).abs() < 1e-14);
///
/// // Pₙ(1) = 1 for all n
/// assert!((legendre(10, 1.0) - 1.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn legendre(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }

    // Three-term recurrence: (n+1)P_{n+1} = (2n+1)x·P_n - n·P_{n-1}
    let mut p_prev = 1.0; // P₀
    let mut p_curr = x; // P₁

    for k in 1..n {
        let k_f64 = k as f64;
        let p_next = ((2.0 * k_f64 + 1.0) * x * p_curr - k_f64 * p_prev) / (k_f64 + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    p_curr
}

/// Compute Legendre polynomials P₀(x) through Pₙ(x).
///
/// Returns all orders from 0 to n (inclusive).
///
/// # Arguments
///
/// * `n` - Maximum polynomial order
/// * `x` - Evaluation point
///
/// # Returns
///
/// Vector of length n+1 containing [P₀(x), P₁(x), ..., Pₙ(x)].
///
/// # Examples
///
/// ```
/// use barracuda::special::legendre::legendre_all;
///
/// let p = legendre_all(3, 0.5);
/// assert_eq!(p.len(), 4);
/// assert!((p[0] - 1.0).abs() < 1e-14);     // P₀
/// assert!((p[1] - 0.5).abs() < 1e-14);     // P₁
/// assert!((p[2] - (-0.125)).abs() < 1e-14); // P₂
/// ```
#[must_use]
pub fn legendre_all(n: usize, x: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(n + 1);

    if n == 0 {
        result.push(1.0);
        return result;
    }

    result.push(1.0); // P₀
    result.push(x); // P₁

    for k in 1..n {
        let k_f64 = k as f64;
        let p_next = ((2.0 * k_f64 + 1.0) * x * result[k] - k_f64 * result[k - 1]) / (k_f64 + 1.0);
        result.push(p_next);
    }

    result
}

/// Compute the associated Legendre function Pₙᵐ(x).
///
/// Uses the recurrence relation starting from Pₘᵐ.
///
/// # Arguments
///
/// * `n` - Degree (n ≥ |m|)
/// * `m` - Order (can be negative)
/// * `x` - Evaluation point in [-1, 1]
///
/// # Returns
///
/// The value Pₙᵐ(x).
///
/// # Convention
///
/// Uses the Condon-Shortley phase convention (includes (-1)ᵐ factor).
///
/// # Examples
///
/// ```
/// use barracuda::special::assoc_legendre;
///
/// // P₁⁰(x) = P₁(x) = x
/// assert!((assoc_legendre(1, 0, 0.5) - 0.5).abs() < 1e-14);
///
/// // P₁¹(x) = -(1-x²)^(1/2)
/// assert!((assoc_legendre(1, 1, 0.5) - (-0.8660254037844386)).abs() < 1e-10);
///
/// // P₂⁰(x) = P₂(x) = (3x²-1)/2
/// assert!((assoc_legendre(2, 0, 0.5) - (-0.125)).abs() < 1e-14);
/// ```
#[expect(clippy::manual_is_multiple_of, reason = "suppressed")] // is_multiple_of is nightly-only
#[must_use]
pub fn assoc_legendre(n: usize, m: i32, x: f64) -> f64 {
    let m_abs = m.unsigned_abs() as usize;

    if m_abs > n {
        return 0.0;
    }

    // Handle m = 0 case (regular Legendre)
    if m == 0 {
        return legendre(n, x);
    }

    // Compute P_m^m first (starting point)
    let sin_theta = (1.0 - x * x).sqrt();
    let mut pmm = 1.0;
    let mut fact = 1.0;

    for _i in 1..=m_abs {
        pmm *= -fact * sin_theta;
        fact += 2.0;
    }

    if n == m_abs {
        return if m < 0 {
            // P_n^{-m} = (-1)^m (n-m)!/(n+m)! P_n^m
            let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
            let factor = factorial_ratio(n, m_abs);
            sign * factor * pmm
        } else {
            pmm
        };
    }

    // Compute P_{m+1}^m
    let mut pmmp1 = x * (2 * m_abs + 1) as f64 * pmm;

    if n == m_abs + 1 {
        return if m < 0 {
            let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
            let factor = factorial_ratio(n, m_abs);
            sign * factor * pmmp1
        } else {
            pmmp1
        };
    }

    // Recurrence for P_l^m from P_{l-1}^m and P_{l-2}^m
    let mut pnm = 0.0;
    for l in (m_abs + 2)..=n {
        let l_f64 = l as f64;
        let m_f64 = m_abs as f64;
        pnm = (x * (2.0 * l_f64 - 1.0) * pmmp1 - (l_f64 + m_f64 - 1.0) * pmm) / (l_f64 - m_f64);
        pmm = pmmp1;
        pmmp1 = pnm;
    }

    if m < 0 {
        let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
        let factor = factorial_ratio(n, m_abs);
        sign * factor * pnm
    } else {
        pnm
    }
}

/// Helper: compute (n-m)! / (n+m)! efficiently
fn factorial_ratio(n: usize, m: usize) -> f64 {
    // (n-m)! / (n+m)! = 1 / [(n-m+1)(n-m+2)...(n+m)]
    let mut result = 1.0;
    for k in (n - m + 1)..=(n + m) {
        result /= k as f64;
    }
    result
}

/// Compute Legendre polynomial for a batch of x values (CPU path).
#[must_use]
pub fn legendre_batch(n: usize, x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| legendre(n, xi)).collect()
}

/// Compute associated Legendre function for a batch of x values.
#[must_use]
pub fn assoc_legendre_batch(n: usize, m: i32, x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| assoc_legendre(n, m, xi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // P_n tests
    #[test]
    fn test_legendre_p0() {
        // P₀(x) = 1
        assert!((legendre(0, 0.0) - 1.0).abs() < 1e-14);
        assert!((legendre(0, 0.5) - 1.0).abs() < 1e-14);
        assert!((legendre(0, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p1() {
        // P₁(x) = x
        assert!((legendre(1, 0.0) - 0.0).abs() < 1e-14);
        assert!((legendre(1, 0.5) - 0.5).abs() < 1e-14);
        assert!((legendre(1, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p2() {
        // P₂(x) = (3x² - 1)/2
        assert!((legendre(2, 0.0) - (-0.5)).abs() < 1e-14);
        assert!((legendre(2, 0.5) - (-0.125)).abs() < 1e-14);
        assert!((legendre(2, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_p3() {
        // P₃(x) = (5x³ - 3x)/2
        assert!((legendre(3, 0.0) - 0.0).abs() < 1e-14);
        let p3_half = (5.0 * 0.125 - 1.5) / 2.0; // (5/8 - 3/2)/2 = -7/16
        assert!((legendre(3, 0.5) - p3_half).abs() < 1e-14);
    }

    #[test]
    fn test_legendre_at_one() {
        // Pₙ(1) = 1 for all n
        for n in 0..10 {
            assert!((legendre(n, 1.0) - 1.0).abs() < 1e-13);
        }
    }

    #[test]
    fn test_legendre_at_minus_one() {
        // Pₙ(-1) = (-1)ⁿ
        for n in 0..10 {
            let expected = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((legendre(n, -1.0) - expected).abs() < 1e-13);
        }
    }

    #[test]
    fn test_legendre_parity() {
        // Pₙ(-x) = (-1)ⁿ Pₙ(x)
        let x = 0.3;
        for n in 0..8 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!((legendre(n, -x) - sign * legendre(n, x)).abs() < 1e-13);
        }
    }

    #[test]
    fn test_legendre_explicit_values() {
        // P₅(x) = (1/8)(63x⁵ - 70x³ + 15x)
        // P₅(0.3) = (0.15309 - 1.89 + 4.5)/8 = 0.345386...
        assert!((legendre(5, 0.3) - 0.345386).abs() < 1e-5);
        // P₁₀(0.5) via recurrence
        assert!((legendre(10, 0.5) - (-0.1882286071777344)).abs() < 1e-10);
    }

    #[test]
    fn test_legendre_all() {
        let p = legendre_all(3, 0.5);
        assert_eq!(p.len(), 4);
        assert!((p[0] - 1.0).abs() < 1e-14);
        assert!((p[1] - 0.5).abs() < 1e-14);
        assert!((p[2] - (-0.125)).abs() < 1e-14);
    }

    // Associated Legendre tests
    #[test]
    fn test_assoc_legendre_m0() {
        // P_n^0(x) = P_n(x)
        for n in 0..5 {
            for x in [0.0, 0.3, 0.5, 0.8, 1.0] {
                assert!((assoc_legendre(n, 0, x) - legendre(n, x)).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn test_assoc_legendre_p11() {
        // P₁¹(x) = -(1-x²)^(1/2)
        let x: f64 = 0.5;
        let expected = -(1.0 - x * x).sqrt();
        assert!((assoc_legendre(1, 1, x) - expected).abs() < 1e-13);
    }

    #[test]
    fn test_assoc_legendre_p21() {
        // P₂¹(x) = -3x(1-x²)^(1/2)
        let x: f64 = 0.5;
        let expected = -3.0 * x * (1.0 - x * x).sqrt();
        assert!((assoc_legendre(2, 1, x) - expected).abs() < 1e-13);
    }

    #[test]
    fn test_assoc_legendre_p22() {
        // P₂²(x) = 3(1-x²)
        let x = 0.5;
        let expected = 3.0 * (1.0 - x * x);
        assert!((assoc_legendre(2, 2, x) - expected).abs() < 1e-13);
    }

    #[test]
    fn test_assoc_legendre_recurrence() {
        // P₃²(x) via recurrence: x(2m+1)·P₂²(x) = 0.5 * 5 * 3(1-0.25) = 5.625
        // Note: This uses the standard recurrence definition, not scipy's lpmv
        // which uses a different phase/scaling convention
        assert!((assoc_legendre(3, 2, 0.5) - 5.625).abs() < 1e-7);
    }

    #[test]
    fn test_assoc_legendre_m_greater_n() {
        // P_n^m = 0 when |m| > n
        assert_eq!(assoc_legendre(2, 3, 0.5), 0.0);
        assert_eq!(assoc_legendre(1, 5, 0.5), 0.0);
    }

    #[test]
    fn test_legendre_batch() {
        let x = vec![0.0, 0.5, 1.0];
        let p2 = legendre_batch(2, &x);
        assert_eq!(p2.len(), 3);
        assert!((p2[0] - (-0.5)).abs() < 1e-14);
        assert!((p2[1] - (-0.125)).abs() < 1e-14);
        assert!((p2[2] - 1.0).abs() < 1e-14);
    }
}
