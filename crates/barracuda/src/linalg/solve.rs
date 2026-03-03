// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear system solvers

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::ops::linalg::LinSolveF64;
use std::sync::Arc;

/// Solve Ax = b using GPU Gaussian elimination with partial pivoting (f64)
///
/// Dispatches to `linsolve_f64.wgsl` shader. All production math uses GPU.
///
/// # Arguments
///
/// * `device` - GPU device (`Arc<WgpuDevice>`)
/// * `a` - Coefficient matrix (row-major, n×n)
/// * `b` - Right-hand side vector (length n)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// Solution vector x, or error if matrix is singular
///
/// # Examples
///
/// ```no_run
/// use barracuda::linalg::solve_f64;
/// use barracuda::prelude::WgpuDevice;
/// use std::sync::Arc;
///
/// # async fn example() -> barracuda::error::Result<()> {
/// let device = Arc::new(WgpuDevice::new().await?);
/// let a = vec![2.0, 1.0, 1.0, 3.0];
/// let b = vec![5.0, 8.0];
/// let x = solve_f64(device, &a, &b, 2)?;
/// # Ok(())
/// # }
/// ```
pub fn solve_f64(device: Arc<WgpuDevice>, a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    if a.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Matrix size mismatch: expected {}×{} = {}, got {}",
                n,
                n,
                n * n,
                a.len()
            ),
        });
    }

    if b.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vector size mismatch: expected {}, got {}", n, b.len()),
        });
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    let solver = LinSolveF64::new(device);
    solver.solve(a, b, n)
}

/// Solve Ax = b on CPU using Gaussian elimination with partial pivoting (f64).
///
/// Use this for **small matrices** (e.g. ESN reservoir sizes 50–200) when a GPU device
/// is unavailable or when the overhead of GPU dispatch outweighs the benefit. For
/// matrices larger than ~500×500, prefer [`solve_f64`] with a GPU device.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (row-major, n×n)
/// * `b` - Right-hand side vector (length n)
/// * `n` - Matrix dimension
///
/// # Returns
///
/// Solution vector x, or error if matrix is singular or inputs are invalid.
pub fn solve_f64_cpu(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    if a.len() != n * n {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Matrix size mismatch: expected {}×{} = {}, got {}",
                n,
                n,
                n * n,
                a.len()
            ),
        });
    }

    if b.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("Vector size mismatch: expected {}, got {}", n, b.len()),
        });
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    // Create augmented matrix [A | b]
    let mut aug = vec![0.0; n * (n + 1)];

    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot row (max |a[i,k]| for i >= k)
        let mut max_row = k;
        let mut max_val = aug[k * (n + 1) + k].abs();

        for i in (k + 1)..n {
            let val = aug[i * (n + 1) + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singularity
        if max_val < 1e-14 {
            return Err(BarracudaError::ExecutionError {
                message: format!("Singular matrix: pivot at column {k} is near-zero ({max_val:e})"),
            });
        }

        // Swap rows k and max_row (full row swap)
        if max_row != k {
            for j in 0..=n {
                aug.swap(k * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        // Eliminate column k below diagonal
        for i in (k + 1)..n {
            let factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];

            for j in k..=n {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    // Backward substitution
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];

        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }

        x[i] = sum / aug[i * (n + 1) + i];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available_sync;

    fn device() -> Option<Arc<WgpuDevice>> {
        get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_solve_f64_2x2() {
        let Some(dev) = device() else { return };
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 8.0];
        let x = solve_f64(dev, &a, &b, 2).unwrap();
        assert!((x[0] - 1.4).abs() < 1e-10);
        assert!((x[1] - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_solve_f64_3x3() {
        // 3x + 2y - z = 1
        // 2x - 2y + 4z = -2
        // -x + 0.5y - z = 0
        let a = vec![3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0];
        let b = vec![1.0, -2.0, 0.0];

        let Some(dev) = device() else { return };
        let x = solve_f64(dev, &a, &b, 3).unwrap();

        // Solution: x=1, y=-2, z=-2
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - (-2.0)).abs() < 1e-10);
        assert!((x[2] - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solve_f64_identity() {
        // Ix = b should give x = b
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 7.0, 9.0];

        let Some(dev) = device() else { return };
        let x = solve_f64(dev, &a, &b, 3).unwrap();

        assert!((x[0] - 5.0).abs() < 1e-14);
        assert!((x[1] - 7.0).abs() < 1e-14);
        assert!((x[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_solve_f64_diagonal() {
        // Diagonal matrix
        let a = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let b = vec![6.0, 9.0, 12.0];

        let Some(dev) = device() else { return };
        let x = solve_f64(dev, &a, &b, 3).unwrap();

        assert!((x[0] - 3.0).abs() < 1e-14);
        assert!((x[1] - 3.0).abs() < 1e-14);
        assert!((x[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_solve_f64_singular_matrix() {
        // GPU LinSolveF64 returns zeros for singular; CPU detects and returns Err.
        // Test CPU path (solve_f64_cpu) for singularity detection.
        let a = vec![1.0, 2.0, 1.0, 2.0];
        let b = vec![1.0, 1.0];
        let result = solve_f64_cpu(&a, &b, 2);
        assert!(result.is_err());
    }

    // ─── solve_f64_cpu tests (no GPU required) ─────────────────────────────

    #[test]
    fn test_solve_f64_cpu_identity() {
        // Ix = b should give x = b
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 7.0, 9.0];
        let x = solve_f64_cpu(&a, &b, 3).expect("identity solve should succeed");
        assert!((x[0] - 5.0).abs() < 1e-14);
        assert!((x[1] - 7.0).abs() < 1e-14);
        assert!((x[2] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_solve_f64_cpu_known_solution() {
        // 2x + y = 5, x + 3y = 8  =>  x = 1.4, y = 2.2
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 8.0];
        let x = solve_f64_cpu(&a, &b, 2).expect("2x2 solve should succeed");
        assert!((x[0] - 1.4).abs() < 1e-10);
        assert!((x[1] - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_solve_f64_cpu_singular_detection() {
        // Singular matrix: rows are linearly dependent
        let a = vec![1.0, 2.0, 1.0, 2.0];
        let b = vec![1.0, 1.0];
        let result = solve_f64_cpu(&a, &b, 2);
        assert!(result.is_err(), "singular matrix should return error");
    }

    #[test]
    fn test_solve_f64_cpu_empty() {
        let x = solve_f64_cpu(&[], &[], 0).expect("empty solve should succeed");
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_solve_f64_cpu_size_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = solve_f64_cpu(&a, &b, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_f64_size_mismatch() {
        let Some(dev) = device() else { return };
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        let result = solve_f64(dev, &a, &b, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_f64_empty() {
        let Some(dev) = device() else { return };
        let x = solve_f64(dev, &[], &[], 0).unwrap();
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn test_solve_f64_large_well_conditioned() {
        let Some(dev) = device() else { return };
        let n = 5;
        let mut a = vec![0.0; n * n];
        let mut b = vec![0.0; n];
        for i in 0..n {
            a[i * n + i] = 10.0;
            b[i] = (i + 1) as f64;
            for j in 0..n {
                if i != j {
                    a[i * n + j] = 0.1;
                }
            }
        }
        let x = solve_f64(dev, &a, &b, n).unwrap();

        // Verify Ax = b
        for i in 0..n {
            let mut ax_i = 0.0;
            for j in 0..n {
                ax_i += a[i * n + j] * x[j];
            }
            assert!(
                (ax_i - b[i]).abs() < 1e-10,
                "Row {}: Ax = {}, b = {}",
                i,
                ax_i,
                b[i]
            );
        }
    }
}
