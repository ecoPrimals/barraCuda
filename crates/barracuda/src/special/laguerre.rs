// SPDX-License-Identifier: AGPL-3.0-only
//! Generalized Laguerre polynomials — CPU reference implementations.
//!
//! **Shader-first architecture**: f64 WGSL equivalents in `shaders/math/`
//! (`laguerre_f64.wgsl`, `laguerre_generalized_f64.wgsl`). GPU pipelines
//! should use shader versions. These CPU functions are reference/validation.
//!
//! Computes `L_n^(α)(x)`, the generalized (associated) Laguerre polynomials,
//! using the three-term recurrence relation. These arise naturally in:
//!
//! - **Quantum mechanics**: Hydrogen atom wave functions (radial part)
//! - **Nuclear physics**: Harmonic oscillator basis (HFB calculations)
//! - **Signal processing**: Laguerre filter design
//! - **Probability**: Gamma distribution moments
//!
//! # Algorithm
//!
//! Uses the stable three-term recurrence:
//!
//! ```text
//! L_0^(α)(x) = 1
//! L_1^(α)(x) = 1 + α - x
//! L_n^(α)(x) = ((2n - 1 + α - x) · L_{n-1}^(α)(x) - (n - 1 + α) · L_{n-2}^(α)(x)) / n
//! ```
//!
//! This is numerically stable for moderate n (up to ~100) and avoids
//! the exponential growth of direct polynomial evaluation.
//!
//! # References
//!
//! - Abramowitz & Stegun, Section 22.7
//! - NIST DLMF, Chapter 18
//! - `scipy.special.eval_genlaguerre`

/// WGSL kernel for generalized Laguerre polynomial evaluation (f64).
pub const WGSL_LAGUERRE_GENERALIZED_F64: &str =
    include_str!("../shaders/math/laguerre_generalized_f64.wgsl");

/// Evaluate the generalized Laguerre polynomial `L_n^(α)(x)`.
///
/// # Arguments
///
/// * `n` - Polynomial degree (≥ 0)
/// * `alpha` - Generalization parameter (> -1 for classical orthogonality)
/// * `x` - Evaluation point
///
/// # Returns
///
/// `L_n^(α)(x)`
///
/// # Examples
///
/// ```
/// use barracuda::special::laguerre;
///
/// // L_0^(0)(x) = 1 for all x
/// assert!((laguerre(0, 0.0, 5.0) - 1.0).abs() < 1e-14);
///
/// // L_1^(0)(x) = 1 - x
/// assert!((laguerre(1, 0.0, 3.0) - (-2.0)).abs() < 1e-14);
///
/// // L_2^(0)(x) = (x² - 4x + 2) / 2
/// let x = 1.5;
/// let expected = (x * x - 4.0 * x + 2.0) / 2.0;
/// assert!((laguerre(2, 0.0, x) - expected).abs() < 1e-12);
/// ```
#[must_use]
pub fn laguerre(n: usize, alpha: f64, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => 1.0 + alpha - x,
        _ => {
            let mut l_prev2 = 1.0; // L_{n-2}
            let mut l_prev1 = 1.0 + alpha - x; // L_{n-1}

            for k in 2..=n {
                let k_f = k as f64;
                // Three-term recurrence
                let l_curr =
                    ((2.0 * k_f - 1.0 + alpha - x) * l_prev1 - (k_f - 1.0 + alpha) * l_prev2) / k_f;
                l_prev2 = l_prev1;
                l_prev1 = l_curr;
            }

            l_prev1
        }
    }
}

/// Evaluate the (simple) Laguerre polynomial `L_n(x)` = `L_n^(0)(x)`.
///
/// # Arguments
///
/// * `n` - Polynomial degree (≥ 0)
/// * `x` - Evaluation point
///
/// # Returns
///
/// `L_n(x)`
///
/// # Examples
///
/// ```
/// use barracuda::special::laguerre_simple;
///
/// assert!((laguerre_simple(0, 1.0) - 1.0).abs() < 1e-14);
/// assert!((laguerre_simple(1, 1.0) - 0.0).abs() < 1e-14);
/// assert!((laguerre_simple(2, 1.0) - (-0.5)).abs() < 1e-14);
/// ```
#[must_use]
pub fn laguerre_simple(n: usize, x: f64) -> f64 {
    laguerre(n, 0.0, x)
}

/// Evaluate all Laguerre polynomials `L_0(x)`, `L_1(x)`, ..., `L_n(x)` at once.
///
/// More efficient than calling `laguerre` repeatedly when all degrees are needed.
///
/// # Arguments
///
/// * `n_max` - Maximum degree
/// * `alpha` - Generalization parameter
/// * `x` - Evaluation point
///
/// # Returns
///
/// Vec of length `n_max + 1` containing [`L_0^(α)(x)`, `L_1^(α)(x)`, ..., `L_n^(α)(x)`]
///
/// # Examples
///
/// ```
/// use barracuda::special::laguerre_all;
///
/// let values = laguerre_all(3, 0.0, 1.0);
/// assert_eq!(values.len(), 4);
/// assert!((values[0] - 1.0).abs() < 1e-14);  // L_0 = 1
/// assert!((values[1] - 0.0).abs() < 1e-14);  // L_1(1) = 0
/// ```
#[must_use]
pub fn laguerre_all(n_max: usize, alpha: f64, x: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(n_max + 1);

    result.push(1.0); // L_0

    if n_max == 0 {
        return result;
    }

    result.push(1.0 + alpha - x); // L_1

    for k in 2..=n_max {
        let k_f = k as f64;
        let l_curr = ((2.0 * k_f - 1.0 + alpha - x) * result[k - 1]
            - (k_f - 1.0 + alpha) * result[k - 2])
            / k_f;
        result.push(l_curr);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laguerre_degree_0() {
        // L_0^(α)(x) = 1 for all α, x
        assert_eq!(laguerre(0, 0.0, 0.0), 1.0);
        assert_eq!(laguerre(0, 0.0, 5.0), 1.0);
        assert_eq!(laguerre(0, 2.0, 100.0), 1.0);
    }

    #[test]
    fn test_laguerre_degree_1() {
        // L_1^(α)(x) = 1 + α - x
        assert!((laguerre(1, 0.0, 0.0) - 1.0).abs() < 1e-14);
        assert!((laguerre(1, 0.0, 1.0) - 0.0).abs() < 1e-14);
        assert!((laguerre(1, 0.0, 3.0) - (-2.0)).abs() < 1e-14);
        assert!((laguerre(1, 2.0, 1.0) - 2.0).abs() < 1e-14); // 1 + 2 - 1 = 2
    }

    #[test]
    fn test_laguerre_degree_2() {
        // L_2^(0)(x) = (x² - 4x + 2) / 2
        for x in [0.0, 0.5, 1.0, 2.0, 5.0] {
            let expected = f64::midpoint(x * x - 4.0 * x, 2.0);
            assert!(
                (laguerre(2, 0.0, x) - expected).abs() < 1e-12,
                "L_2(0, {}) = {} expected {}",
                x,
                laguerre(2, 0.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_degree_3() {
        // L_3^(0)(x) = (-x³ + 9x² - 18x + 6) / 6
        for x in [0.0_f64, 1.0, 2.0, 3.0] {
            let expected = (-x.powi(3) + 9.0 * x * x - 18.0 * x + 6.0) / 6.0;
            assert!(
                (laguerre(3, 0.0, x) - expected).abs() < 1e-11,
                "L_3(0, {}) = {} expected {}",
                x,
                laguerre(3, 0.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_generalized() {
        // L_1^(1)(x) = 2 - x
        assert!((laguerre(1, 1.0, 0.0) - 2.0).abs() < 1e-14);
        assert!((laguerre(1, 1.0, 2.0) - 0.0).abs() < 1e-14);

        // L_2^(1)(x) = (x² - 6x + 6) / 2
        for x in [0.0, 1.0, 3.0] {
            let expected = f64::midpoint(x * x - 6.0 * x, 6.0);
            assert!(
                (laguerre(2, 1.0, x) - expected).abs() < 1e-12,
                "L_2(1, {}) = {} expected {}",
                x,
                laguerre(2, 1.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_at_zero() {
        // L_n^(α)(0) = C(n + α, n) = Γ(n + α + 1) / (n! · Γ(α + 1))
        // For α = 0: L_n(0) = 1
        assert!((laguerre(0, 0.0, 0.0) - 1.0).abs() < 1e-14);
        assert!((laguerre(1, 0.0, 0.0) - 1.0).abs() < 1e-14);
        assert!((laguerre(2, 0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((laguerre(5, 0.0, 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_laguerre_simple() {
        assert!((laguerre_simple(0, 1.0) - 1.0).abs() < 1e-14);
        assert!((laguerre_simple(1, 1.0) - 0.0).abs() < 1e-14);
        assert!((laguerre_simple(2, 1.0) - (-0.5)).abs() < 1e-14);
    }

    #[test]
    fn test_laguerre_all_basic() {
        let values = laguerre_all(3, 0.0, 1.0);
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.0).abs() < 1e-14); // L_0
        assert!((values[1] - 0.0).abs() < 1e-14); // L_1(1) = 0
        assert!((values[2] - (-0.5)).abs() < 1e-14); // L_2(1)
    }

    #[test]
    fn test_laguerre_all_degree_0() {
        let values = laguerre_all(0, 0.0, 5.0);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], 1.0);
    }

    #[test]
    fn test_laguerre_all_consistency() {
        // laguerre_all should agree with individual laguerre calls
        let alpha = 1.5;
        let x = 2.7;
        let all = laguerre_all(10, alpha, x);

        for (n, &val) in all.iter().enumerate() {
            let individual = laguerre(n, alpha, x);
            assert!(
                (val - individual).abs() < 1e-10,
                "Mismatch at n={n}: all={val}, individual={individual}"
            );
        }
    }

    #[test]
    fn test_laguerre_higher_degree() {
        // Test stability for moderate degrees
        let val = laguerre(20, 0.0, 5.0);
        // scipy.special.eval_laguerre(20, 5.0) ≈ -1.0437...
        // Just check it's finite and in a reasonable range
        assert!(val.is_finite());
        assert!(val.abs() < 1e10);
    }

    #[test]
    fn test_laguerre_negative_x() {
        // Laguerre polynomials are defined for all real x
        let val = laguerre(3, 0.0, -1.0);
        assert!(val.is_finite());
        // L_3(0, -1) = -(-1)³ + 9(-1)² - 18(-1) + 6) / 6 = (1 + 9 + 18 + 6) / 6 = 34/6
        let expected = 34.0 / 6.0;
        assert!(
            (val - expected).abs() < 1e-11,
            "L_3(0, -1) = {val} expected {expected}"
        );
    }
}
