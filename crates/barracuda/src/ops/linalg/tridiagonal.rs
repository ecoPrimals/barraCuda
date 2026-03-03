// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tridiagonal matrix solver (Thomas algorithm)
//!
//! Solves tridiagonal linear systems A·x = d in O(n) time using
//! the Thomas algorithm (modified Gaussian elimination).
//!
//! # System Structure
//!
//! ```text
//! | b₀  c₀  0   0   ...  0  | | x₀ |   | d₀ |
//! | a₁  b₁  c₁  0   ...  0  | | x₁ |   | d₁ |
//! | 0   a₂  b₂  c₂  ...  0  | | x₂ | = | d₂ |
//! | ⋮                    ⋮  | | ⋮  |   | ⋮  |
//! | 0   ...  0  aₙ₋₁ bₙ₋₁ | | xₙ₋₁|   | dₙ₋₁|
//! ```
//!
//! # Applications
//!
//! - **Crank-Nicolson PDE** (TTM heat equation)
//! - **Cubic spline interpolation**
//! - **Implicit finite difference schemes**
//!
//! # References
//!
//! - Numerical Recipes, §2.4
//! - DLMF: <https://dlmf.nist.gov>

use crate::error::{BarracudaError, Result};

/// Solve a tridiagonal system A·x = d using the Thomas algorithm.
///
/// # Arguments
///
/// * `a` - Sub-diagonal (length n-1): a[i] is A[i+1, i]
/// * `b` - Main diagonal (length n): b[i] is A[i, i]
/// * `c` - Super-diagonal (length n-1): c[i] is A[i, i+1]
/// * `d` - Right-hand side (length n)
///
/// # Returns
///
/// Solution vector x of length n.
///
/// # Complexity
///
/// O(n) time, O(n) space for temporary arrays.
///
/// # Examples
///
/// ```
/// use barracuda::ops::linalg::tridiagonal::tridiagonal_solve;
///
/// // Simple 3x3 system
/// let a = vec![1.0, 1.0];           // sub-diagonal
/// let b = vec![4.0, 4.0, 4.0];      // main diagonal
/// let c = vec![1.0, 1.0];           // super-diagonal
/// let d = vec![1.0, 2.0, 1.0];      // RHS
///
/// let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
/// // Verify: A·x ≈ d
/// ```
pub fn tridiagonal_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();

    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "System size must be > 0".to_string(),
        });
    }

    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Dimension mismatch: a.len()={}, b.len()={}, c.len()={}, d.len()={}",
                a.len(),
                b.len(),
                c.len(),
                d.len()
            ),
        });
    }

    // Special case: n=1
    if n == 1 {
        if b[0].abs() < 1e-15 {
            return Err(BarracudaError::Numerical {
                message: "Division by zero: b[0] is near zero".to_string(),
            });
        }
        return Ok(vec![d[0] / b[0]]);
    }

    // Forward sweep: eliminate sub-diagonal
    let mut c_prime = vec![0.0; n - 1];
    let mut d_prime = vec![0.0; n];

    // First row
    if b[0].abs() < 1e-15 {
        return Err(BarracudaError::Numerical {
            message: "Division by zero: b[0] is near zero".to_string(),
        });
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    // Forward sweep for i = 1..n-1
    for i in 1..n {
        let denom = b[i] - a[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-15 {
            return Err(BarracudaError::Numerical {
                message: format!("Division by zero at row {i}: matrix may be singular"),
            });
        }
        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve a tridiagonal system with f32 precision.
///
/// Same as [`tridiagonal_solve`] but uses f32 for GPU compatibility.
pub fn tridiagonal_solve_f32(a: &[f32], b: &[f32], c: &[f32], d: &[f32]) -> Result<Vec<f32>> {
    let n = b.len();

    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "System size must be > 0".to_string(),
        });
    }

    if a.len() != n - 1 || c.len() != n - 1 || d.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Dimension mismatch: a.len()={}, b.len()={}, c.len()={}, d.len()={}",
                a.len(),
                b.len(),
                c.len(),
                d.len()
            ),
        });
    }

    if n == 1 {
        if b[0].abs() < 1e-7 {
            return Err(BarracudaError::Numerical {
                message: "Division by zero: b[0] is near zero".to_string(),
            });
        }
        return Ok(vec![d[0] / b[0]]);
    }

    let mut c_prime = vec![0.0f32; n - 1];
    let mut d_prime = vec![0.0f32; n];

    if b[0].abs() < 1e-7 {
        return Err(BarracudaError::Numerical {
            message: "Division by zero: b[0] is near zero".to_string(),
        });
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-7 {
            return Err(BarracudaError::Numerical {
                message: format!("Division by zero at row {i}: matrix may be singular"),
            });
        }
        if i < n - 1 {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom;
    }

    let mut x = vec![0.0f32; n];
    x[n - 1] = d_prime[n - 1];

    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve multiple tridiagonal systems with the same structure.
///
/// Efficient when solving the same matrix A with multiple right-hand sides.
///
/// # Arguments
///
/// * `a`, `b`, `c` - Matrix diagonals (same for all systems)
/// * `rhs_batch` - Vector of right-hand sides
///
/// # Returns
///
/// Vector of solution vectors.
pub fn tridiagonal_solve_batch(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    rhs_batch: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    rhs_batch
        .iter()
        .map(|d| tridiagonal_solve(a, b, c, d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tridiagonal_simple() {
        // Simple diagonally dominant 3x3 system
        let a = vec![1.0, 1.0];
        let b = vec![4.0, 4.0, 4.0];
        let c = vec![1.0, 1.0];
        let d = vec![5.0, 6.0, 5.0];

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), 3);

        // Verify solution by computing A·x
        let ax0 = b[0] * x[0] + c[0] * x[1];
        let ax1 = a[0] * x[0] + b[1] * x[1] + c[1] * x[2];
        let ax2 = a[1] * x[1] + b[2] * x[2];

        assert!((ax0 - d[0]).abs() < 1e-12);
        assert!((ax1 - d[1]).abs() < 1e-12);
        assert!((ax2 - d[2]).abs() < 1e-12);
    }

    #[test]
    fn test_tridiagonal_identity() {
        // Identity matrix: b=1, a=c=0
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0, 1.0];
        let c = vec![0.0, 0.0];
        let d = vec![1.0, 2.0, 3.0];

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14);
        assert!((x[1] - 2.0).abs() < 1e-14);
        assert!((x[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_tridiagonal_n1() {
        // Single equation
        let a: Vec<f64> = vec![];
        let b = vec![2.0];
        let c: Vec<f64> = vec![];
        let d = vec![4.0];

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), 1);
        assert!((x[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_tridiagonal_n2() {
        // 2x2 system
        let a = vec![1.0];
        let b = vec![2.0, 2.0];
        let c = vec![1.0];
        let d = vec![3.0, 3.0];

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), 2);

        // Verify
        let ax0 = b[0] * x[0] + c[0] * x[1];
        let ax1 = a[0] * x[0] + b[1] * x[1];
        assert!((ax0 - d[0]).abs() < 1e-12);
        assert!((ax1 - d[1]).abs() < 1e-12);
    }

    #[test]
    fn test_tridiagonal_heat_equation() {
        // Heat equation discretization matrix
        // -u_{i-1} + 2u_i - u_{i+1} = f_i
        let n = 10;
        let a = vec![-1.0; n - 1];
        let b = vec![2.0; n];
        let c = vec![-1.0; n - 1];
        let d = vec![1.0; n]; // uniform source

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), n);

        // Verify residual is small
        for i in 0..n {
            let ax_i = if i == 0 {
                b[0] * x[0] + c[0] * x[1]
            } else if i == n - 1 {
                a[n - 2] * x[n - 2] + b[n - 1] * x[n - 1]
            } else {
                a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
            };
            assert!(
                (ax_i - d[i]).abs() < 1e-10,
                "Residual at {}: {}",
                i,
                ax_i - d[i]
            );
        }
    }

    #[test]
    fn test_tridiagonal_f32() {
        let a = vec![1.0f32, 1.0];
        let b = vec![4.0f32, 4.0, 4.0];
        let c = vec![1.0f32, 1.0];
        let d = vec![5.0f32, 6.0, 5.0];

        let x = tridiagonal_solve_f32(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), 3);

        let ax0 = b[0] * x[0] + c[0] * x[1];
        assert!((ax0 - d[0]).abs() < 1e-5);
    }

    #[test]
    fn test_tridiagonal_batch() {
        let a = vec![1.0, 1.0];
        let b = vec![4.0, 4.0, 4.0];
        let c = vec![1.0, 1.0];

        let rhs = vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 4.0, 2.0],
            vec![0.0, 1.0, 0.0],
        ];

        let solutions = tridiagonal_solve_batch(&a, &b, &c, &rhs).unwrap();
        assert_eq!(solutions.len(), 3);

        // Each solution should satisfy A·x = d
        for (sol, d) in solutions.iter().zip(rhs.iter()) {
            let ax0 = b[0] * sol[0] + c[0] * sol[1];
            assert!((ax0 - d[0]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tridiagonal_error_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let c: Vec<f64> = vec![];
        let d: Vec<f64> = vec![];

        assert!(tridiagonal_solve(&a, &b, &c, &d).is_err());
    }

    #[test]
    fn test_tridiagonal_error_mismatch() {
        let a = vec![1.0];
        let b = vec![2.0, 2.0, 2.0]; // n=3
        let c = vec![1.0]; // should be length 2
        let d = vec![1.0, 1.0, 1.0];

        assert!(tridiagonal_solve(&a, &b, &c, &d).is_err());
    }

    #[test]
    fn test_tridiagonal_large() {
        // Larger system
        let n = 100;
        let a = vec![1.0; n - 1];
        let b = vec![4.0; n];
        let c = vec![1.0; n - 1];
        let d: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let x = tridiagonal_solve(&a, &b, &c, &d).unwrap();
        assert_eq!(x.len(), n);

        // Check a few residuals
        let ax0 = b[0] * x[0] + c[0] * x[1];
        assert!((ax0 - d[0]).abs() < 1e-10);
    }
}
