// SPDX-License-Identifier: AGPL-3.0-only
//! Factorial function

/// WGSL kernel for factorial computation (f64).
pub const WGSL_FACTORIAL_F64: &str = include_str!("../shaders/special/factorial_f64.wgsl");

/// Compute n! (factorial)
///
/// Uses exact calculation for n ≤ 20, Stirling's approximation for larger n.
///
/// # Arguments
///
/// * `n` - Non-negative integer
///
/// # Returns
///
/// n! as f64
///
/// # Examples
///
/// ```
/// use barracuda::special::factorial;
///
/// assert_eq!(factorial(0), 1.0);
/// assert_eq!(factorial(1), 1.0);
/// assert_eq!(factorial(5), 120.0);
/// assert_eq!(factorial(10), 3628800.0);
/// ```
///
/// # Algorithm
///
/// - **n ≤ 20**: Exact integer multiplication
/// - **n > 20**: Stirling's approximation n! ≈ √(2πn) · (n/e)ⁿ
///
/// # Precision
///
/// - Exact for n ≤ 20
/// - Stirling's approximation has <1% error for n > 20
#[must_use]
pub fn factorial(n: usize) -> f64 {
    // Lookup table for exact values (n ≤ 20)
    const FACTORIAL_TABLE: [f64; 21] = [
        1.0,                   // 0!
        1.0,                   // 1!
        2.0,                   // 2!
        6.0,                   // 3!
        24.0,                  // 4!
        120.0,                 // 5!
        720.0,                 // 6!
        5040.0,                // 7!
        40320.0,               // 8!
        362_880.0,             // 9!
        3628800.0,             // 10!
        39916800.0,            // 11!
        479001600.0,           // 12!
        6227020800.0,          // 13!
        87178291200.0,         // 14!
        1307674368000.0,       // 15!
        20922789888000.0,      // 16!
        355687428096000.0,     // 17!
        6402373705728000.0,    // 18!
        121645100408832000.0,  // 19!
        2432902008176640000.0, // 20!
    ];

    if n <= 20 {
        FACTORIAL_TABLE[n]
    } else {
        // Stirling's approximation: n! ≈ √(2πn) · (n/e)ⁿ
        let n_f64 = n as f64;
        let sqrt_2pi_n = (2.0 * std::f64::consts::PI * n_f64).sqrt();
        let n_over_e = n_f64 / std::f64::consts::E;
        sqrt_2pi_n * n_over_e.powf(n_f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_small() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(2), 2.0);
        assert_eq!(factorial(3), 6.0);
        assert_eq!(factorial(4), 24.0);
        assert_eq!(factorial(5), 120.0);
    }

    #[test]
    fn test_factorial_medium() {
        assert_eq!(factorial(10), 3628800.0);
        assert_eq!(factorial(15), 1307674368000.0);
        assert_eq!(factorial(20), 2432902008176640000.0);
    }

    #[test]
    fn test_factorial_large_stirling() {
        // For n > 20, use Stirling's approximation
        // Verify it's close to the pattern
        let f25 = factorial(25);

        // 25! ≈ 1.55e25
        assert!(f25 > 1e25);
        assert!(f25 < 2e25);

        // Verify Stirling is monotonically increasing
        assert!(factorial(30) > factorial(25));
        assert!(factorial(50) > factorial(30));
    }

    #[test]
    fn test_factorial_stirling_accuracy() {
        // Stirling's approximation has ~1% error for n > 20
        // 21! exact = 51090942171709440000 ≈ 5.109e19
        let f21 = factorial(21);
        let expected = 51090942171709440000.0;
        let relative_error = (f21 - expected).abs() / expected;

        // Stirling should be within 1% for n=21
        assert!(
            relative_error < 0.01,
            "Stirling error too large: {} vs {}, error = {:.2}%",
            f21,
            expected,
            relative_error * 100.0
        );
    }
}
