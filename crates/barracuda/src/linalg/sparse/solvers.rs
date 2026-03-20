// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sparse iterative solvers
//!
//! Provides iterative methods for solving sparse linear systems Ax = b.
//!
//! # Methods
//!
//! - **CG** (Conjugate Gradient): Best for symmetric positive definite (SPD)
//! - **`BiCGSTAB`**: For general non-symmetric systems
//! - **Jacobi**: Simple iteration, good for diagonally dominant systems
//! - **Gauss-Seidel**: Faster than Jacobi for many problems
//!
//! # Convergence
//!
//! All methods use a residual-based stopping criterion:
//! ‖b - Ax‖ / ‖b‖ < tolerance
//!
//! # Preconditioning
//!
//! For faster convergence, consider diagonal (Jacobi) preconditioning:
//! - Cheap: O(n) setup, O(nnz) per iteration
//! - Effective for diagonally dominant systems

use super::csr::CsrMatrix;
use crate::error::{BarracudaError, Result};

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Use diagonal preconditioning
    pub use_preconditioner: bool,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_iterations: 1000,
            use_preconditioner: true,
            verbose: false,
        }
    }
}

impl SolverConfig {
    /// Create with custom tolerance
    #[must_use]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Create with custom max iterations
    #[must_use]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Disable preconditioning
    #[must_use]
    pub fn no_preconditioner(mut self) -> Self {
        self.use_preconditioner = false;
        self
    }
}

/// Solver result
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Solution vector
    pub x: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final relative residual
    pub residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

impl SolverResult {
    /// Check if solve was successful
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.converged
    }
}

/// Compute vector dot product
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute vector L2 norm
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Vector: y = a*x + y
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// Vector: y = x - y (y = x + (-1)*y)
fn sub_into(x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = xi - *yi;
    }
}

/// Apply diagonal preconditioner: z = M⁻¹ * r
fn apply_preconditioner(diag: &[f64], r: &[f64], z: &mut [f64]) {
    for (i, (&ri, &di)) in r.iter().zip(diag.iter()).enumerate() {
        // Robust division - avoid very small divisors
        z[i] = if di.abs() > 1e-12 { ri / di } else { ri };
    }
}

/// Conjugate Gradient solver for symmetric positive definite matrices
///
/// Solves Ax = b where A is SPD.
///
/// # Arguments
///
/// * `a` - Symmetric positive definite CSR matrix
/// * `b` - Right-hand side vector
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Example
///
/// ```
/// use barracuda::linalg::sparse::{CsrMatrix, cg_solve};
///
/// // Tridiagonal SPD matrix
/// let a = CsrMatrix::from_triplets(3, 3, &[
///     (0, 0, 4.0), (0, 1, -1.0),
///     (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
///     (2, 1, -1.0), (2, 2, 4.0),
/// ]);
///
/// let b = vec![1.0, 2.0, 3.0];
/// let result = cg_solve(&a, &b, 1e-10, 100).unwrap();
///
/// assert!(result.converged);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square, if `b.len() != n`, or if
/// [`matvec`](super::csr::CsrMatrix::matvec) fails during iteration.
pub fn cg_solve(a: &CsrMatrix, b: &[f64], tol: f64, max_iter: usize) -> Result<SolverResult> {
    let config = SolverConfig::default()
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    cg_solve_with_config(a, b, &config)
}

/// Conjugate Gradient with configuration
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square, if `b.len() != n`, or if [`matvec`](super::csr::CsrMatrix::matvec)
/// fails during iteration.
pub fn cg_solve_with_config(
    a: &CsrMatrix,
    b: &[f64],
    config: &SolverConfig,
) -> Result<SolverResult> {
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(BarracudaError::InvalidInput {
            message: "CG requires square matrix".to_string(),
        });
    }
    if b.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vector length {} doesn't match matrix size {}", b.len(), n),
        });
    }

    // Initial guess: x = 0
    let mut x = vec![0.0; n];

    // r = b - A*x = b (since x = 0)
    let mut r = b.to_vec();
    let b_norm = norm(b);

    if b_norm < 1e-14 {
        return Ok(SolverResult {
            x,
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    // Diagonal preconditioner
    let diag = if config.use_preconditioner {
        a.diagonal()
    } else {
        vec![1.0; n]
    };

    // z = M⁻¹ * r
    let mut z = vec![0.0; n];
    apply_preconditioner(&diag, &r, &mut z);

    // p = z
    let mut p = z.clone();

    // rz = r·z
    let mut rz = dot(&r, &z);

    for iter in 0..config.max_iterations {
        // Ap = A * p
        let ap = a.matvec(&p)?;

        // α = rz / (p·Ap)
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-30 {
            // Near-breakdown - return current solution
            let residual = norm(&r) / b_norm;
            return Ok(SolverResult {
                x,
                iterations: iter + 1,
                residual,
                converged: residual < config.tolerance,
            });
        }
        let alpha = rz / pap;

        // x = x + α*p
        axpy(alpha, &p, &mut x);

        // r = r - α*Ap
        axpy(-alpha, &ap, &mut r);

        // Check convergence
        let residual = norm(&r) / b_norm;
        if residual < config.tolerance {
            return Ok(SolverResult {
                x,
                iterations: iter + 1,
                residual,
                converged: true,
            });
        }

        // z = M⁻¹ * r
        apply_preconditioner(&diag, &r, &mut z);

        // β = (r·z)_new / (r·z)_old
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        // p = z + β*p
        for (pi, &zi) in p.iter_mut().zip(z.iter()) {
            *pi = zi + beta * *pi;
        }
    }

    // Did not converge
    Ok(SolverResult {
        x,
        iterations: config.max_iterations,
        residual: norm(&r) / b_norm,
        converged: false,
    })
}

/// `BiCGSTAB` solver for general non-symmetric matrices
///
/// Solves Ax = b where A is any square matrix.
///
/// # Arguments
///
/// * `a` - Square CSR matrix
/// * `b` - Right-hand side vector
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Example
///
/// ```
/// use barracuda::linalg::sparse::{CsrMatrix, bicgstab_solve};
///
/// // Non-symmetric matrix
/// let a = CsrMatrix::from_triplets(3, 3, &[
///     (0, 0, 4.0), (0, 1, 1.0),
///     (1, 0, -1.0), (1, 1, 3.0), (1, 2, 1.0),
///     (2, 1, -1.0), (2, 2, 2.0),
/// ]);
///
/// let b = vec![1.0, 2.0, 3.0];
/// let result = bicgstab_solve(&a, &b, 1e-10, 100).unwrap();
///
/// assert!(result.converged);
/// ```
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square, if `b.len() != n`, if [`matvec`](super::csr::CsrMatrix::matvec)
/// fails, or if `BiCGSTAB` breaks down (rho = 0, `r_hat·v` = 0, or omega = 0).
pub fn bicgstab_solve(a: &CsrMatrix, b: &[f64], tol: f64, max_iter: usize) -> Result<SolverResult> {
    let config = SolverConfig::default()
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    bicgstab_solve_with_config(a, b, &config)
}

/// `BiCGSTAB` with configuration
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square, if `b.len() != n`, if [`matvec`](super::csr::CsrMatrix::matvec)
/// fails, or if `BiCGSTAB` breaks down (rho = 0, `r_hat·v` = 0, or omega = 0).
pub fn bicgstab_solve_with_config(
    a: &CsrMatrix,
    b: &[f64],
    config: &SolverConfig,
) -> Result<SolverResult> {
    let n = a.n_rows;
    if a.n_cols != n {
        return Err(BarracudaError::InvalidInput {
            message: "BiCGSTAB requires square matrix".to_string(),
        });
    }
    if b.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vector length {} doesn't match matrix size {}", b.len(), n),
        });
    }

    // Initial guess: x = 0
    let mut x = vec![0.0; n];

    // r = b - A*x = b
    let mut r = b.to_vec();
    let b_norm = norm(b);

    if b_norm < 1e-14 {
        return Ok(SolverResult {
            x,
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    // r_hat = r (shadow residual, fixed)
    let r_hat = r.clone();

    // Initialize
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;

    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];

    // Diagonal preconditioner
    let diag = if config.use_preconditioner {
        a.diagonal()
    } else {
        vec![1.0; n]
    };

    for iter in 0..config.max_iterations {
        // ρ = r_hat · r
        let rho_new = dot(&r_hat, &r);

        if rho_new.abs() < 1e-14 {
            return Err(BarracudaError::Internal(
                "BiCGSTAB breakdown: rho = 0".to_string(),
            ));
        }

        // β = (ρ_new / ρ) * (α / ω)
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + β * (p - ω * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply preconditioner to p
        let mut p_hat = vec![0.0; n];
        apply_preconditioner(&diag, &p, &mut p_hat);

        // v = A * p_hat
        v = a.matvec(&p_hat)?;

        // α = ρ / (r_hat · v)
        let rv = dot(&r_hat, &v);
        if rv.abs() < 1e-14 {
            return Err(BarracudaError::Internal(
                "BiCGSTAB breakdown: r_hat·v = 0".to_string(),
            ));
        }
        alpha = rho / rv;

        // s = r - α * v
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        // Check convergence after first half-step
        let s_norm = norm(&s);
        if s_norm / b_norm < config.tolerance {
            // x = x + α * p_hat
            axpy(alpha, &p_hat, &mut x);
            return Ok(SolverResult {
                x,
                iterations: iter + 1,
                residual: s_norm / b_norm,
                converged: true,
            });
        }

        // Apply preconditioner to s
        let mut s_hat = vec![0.0; n];
        apply_preconditioner(&diag, &s, &mut s_hat);

        // t = A * s_hat
        t = a.matvec(&s_hat)?;

        // ω = (t · s) / (t · t)
        let tt = dot(&t, &t);
        if tt.abs() < 1e-14 {
            omega = 0.0;
        } else {
            omega = dot(&t, &s) / tt;
        }

        // x = x + α * p_hat + ω * s_hat
        axpy(alpha, &p_hat, &mut x);
        axpy(omega, &s_hat, &mut x);

        // r = s - ω * t
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }

        // Check convergence
        let residual = norm(&r) / b_norm;
        if residual < config.tolerance {
            return Ok(SolverResult {
                x,
                iterations: iter + 1,
                residual,
                converged: true,
            });
        }

        if omega.abs() < 1e-14 {
            return Err(BarracudaError::Internal(
                "BiCGSTAB breakdown: omega = 0".to_string(),
            ));
        }
    }

    // Did not converge
    Ok(SolverResult {
        x,
        iterations: config.max_iterations,
        residual: norm(&r) / b_norm,
        converged: false,
    })
}

/// Jacobi iteration solver
///
/// Simple iterative method for diagonally dominant systems.
///
/// # Arguments
///
/// * `a` - Square CSR matrix (should be diagonally dominant)
/// * `b` - Right-hand side vector
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Errors
///
/// Returns [`Err`] if the matrix is not square or `b.len() != n` (dimension mismatch), if any
/// diagonal element is zero, or if [`matvec`](super::csr::CsrMatrix::matvec) fails during iteration.
pub fn jacobi_solve(a: &CsrMatrix, b: &[f64], tol: f64, max_iter: usize) -> Result<SolverResult> {
    let n = a.n_rows;
    if a.n_cols != n || b.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: "Dimension mismatch".to_string(),
        });
    }

    let b_norm = norm(b);
    if b_norm < 1e-14 {
        return Ok(SolverResult {
            x: vec![0.0; n],
            iterations: 0,
            residual: 0.0,
            converged: true,
        });
    }

    // Get diagonal
    let diag = a.diagonal();

    // Check diagonal is non-zero
    for (i, d) in diag.iter().enumerate() {
        if d.abs() < 1e-14 {
            return Err(BarracudaError::InvalidInput {
                message: format!("Zero diagonal at row {i}"),
            });
        }
    }

    let mut x = vec![0.0; n];
    let mut x_new = vec![0.0; n];

    for iter in 0..max_iter {
        // x_new[i] = (b[i] - Σ_{j≠i} A[i,j] * x[j]) / A[i,i]
        for i in 0..n {
            let row_start = a.row_ptr[i];
            let row_end = a.row_ptr[i + 1];

            let mut sum = b[i];
            for k in row_start..row_end {
                let j = a.col_indices[k];
                if j != i {
                    sum -= a.values[k] * x[j];
                }
            }
            x_new[i] = sum / diag[i];
        }

        // Check convergence
        let ax = a.matvec(&x_new)?;
        let mut r = ax;
        sub_into(b, &mut r);
        let residual = norm(&r) / b_norm;

        if residual < tol {
            return Ok(SolverResult {
                x: x_new,
                iterations: iter + 1,
                residual,
                converged: true,
            });
        }

        std::mem::swap(&mut x, &mut x_new);
    }

    // Did not converge
    let ax = a.matvec(&x)?;
    let mut r = ax;
    sub_into(b, &mut r);

    Ok(SolverResult {
        x,
        iterations: max_iter,
        residual: norm(&r) / b_norm,
        converged: false,
    })
}

#[cfg(test)]
#[path = "solvers_tests.rs"]
mod tests;
