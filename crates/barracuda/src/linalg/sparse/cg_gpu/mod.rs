// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Conjugate Gradient Solver (f64)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Full f64 precision via SPIR-V/Vulkan (bypasses CUDA fp64 throttle)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//!
//! ## Algorithm
//!
//! Preconditioned Conjugate Gradient for symmetric positive definite systems:
//! ```text
//! x₀ = 0, r₀ = b, z₀ = M⁻¹r₀, p₀ = z₀
//! For k = 0, 1, ...
//!   αₖ = (rₖ·zₖ) / (pₖ·Apₖ)
//!   xₖ₊₁ = xₖ + αₖpₖ
//!   rₖ₊₁ = rₖ - αₖApₖ
//!   Check convergence: ‖rₖ₊₁‖ / ‖b‖ < tol
//!   zₖ₊₁ = M⁻¹rₖ₊₁
//!   βₖ = (rₖ₊₁·zₖ₊₁) / (rₖ·zₖ)
//!   pₖ₊₁ = zₖ₊₁ + βₖpₖ
//! ```
//!
//! ## Precision
//!
//! **Full f64 precision** - uses native WGSL f64 via SPIR-V/Vulkan.
//! FP64 performance is 1:2-3 (not 1:32 like CUDA consumer GPUs).
//!
//! ## References
//!
//! - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems
//! - Golub & Van Loan, "Matrix Computations"

mod gpu_resident;
mod preconditioned;

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::linalg::sparse::csr::CsrMatrix;
use std::sync::Arc;

/// GPU Conjugate Gradient solver result
#[derive(Debug, Clone)]
pub struct CgGpuResult {
    /// Solution vector
    pub x: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final relative residual
    pub residual: f64,
    /// Whether convergence was achieved
    pub converged: bool,
}

impl CgGpuResult {
    /// Returns true if the solver converged.
    pub fn is_ok(&self) -> bool {
        self.converged
    }
}

/// GPU-accelerated Conjugate Gradient solver
pub struct CgGpu;

impl CgGpu {
    // Separate shader modules to avoid binding conflicts
    fn spmv_shader() -> &'static str {
        include_str!("../../../shaders/sparse/spmv_f64.wgsl")
    }

    fn dot_reduce_shader() -> &'static str {
        include_str!("../../../shaders/sparse/dot_reduce_f64.wgsl")
    }

    fn cg_kernels_shader() -> &'static str {
        include_str!("../../../shaders/sparse/cg_kernels_f64.wgsl")
    }

    /// Solve Ax = b using GPU-accelerated Conjugate Gradient
    ///
    /// Delegates to `solve_gpu_resident` with convergence check every iteration.
    /// For fewer GPU↔CPU syncs on large systems, use `solve_gpu_resident` with larger `check_interval`.
    pub fn solve(
        device: Arc<WgpuDevice>,
        a: &CsrMatrix,
        b: &[f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<CgGpuResult> {
        Self::solve_gpu_resident(device, a, b, tol, max_iter, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::sparse::csr::CsrMatrix;

    fn create_spd_tridiagonal(n: usize) -> CsrMatrix {
        let mut triplets = Vec::new();

        for i in 0..n {
            triplets.push((i, i, 4.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0));
            }
        }

        CsrMatrix::from_triplets(n, n, &triplets)
    }

    #[tokio::test]
    async fn test_cg_gpu_small() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return; // Skip if no f64 GPU
        };

        let a = create_spd_tridiagonal(3);
        let b = vec![1.0, 2.0, 3.0];

        let result = CgGpu::solve(device, &a, &b, 1e-10, 100).unwrap();

        assert!(result.converged, "CG should converge");
        assert!(result.residual < 1e-10, "Residual should be small");

        // Verify: Ax ≈ b
        let ax = a.matvec(&result.x).unwrap();
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - bi).abs() < 1e-8, "Ax should equal b");
        }
    }

    #[tokio::test]
    async fn test_cg_gpu_resident() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return; // Skip if no f64 GPU
        };

        // Test GPU-resident CG with larger system
        let a = create_spd_tridiagonal(100);
        let b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

        // Using check_interval=10 to reduce GPU↔CPU syncs
        let result = CgGpu::solve_gpu_resident(device.clone(), &a, &b, 1e-10, 500, 10).unwrap();

        assert!(result.converged, "GPU-resident CG should converge");
        assert!(
            result.residual < 1e-10,
            "Residual should be small: {}",
            result.residual
        );

        // Verify: Ax ≈ b
        let ax = a.matvec(&result.x).unwrap();
        for (i, (axi, bi)) in ax.iter().zip(b.iter()).enumerate() {
            assert!(
                (axi - bi).abs() < 1e-6,
                "Ax[{}] = {} should equal b[{}] = {}",
                i,
                axi,
                i,
                bi
            );
        }
    }

    #[tokio::test]
    async fn test_cg_gpu_resident_vs_original() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return; // Skip if no f64 GPU
        };

        // Compare GPU-resident vs original implementation
        let a = create_spd_tridiagonal(50);
        let b: Vec<f64> = (0..50).map(|i| ((i + 1) as f64).sin()).collect();

        let result_original = CgGpu::solve(device.clone(), &a, &b, 1e-10, 200).unwrap();
        let result_resident =
            CgGpu::solve_gpu_resident(device.clone(), &a, &b, 1e-10, 200, 5).unwrap();

        // Both should converge
        assert!(result_original.converged, "Original CG should converge");
        assert!(result_resident.converged, "GPU-resident CG should converge");

        // Solutions should be nearly identical
        for (i, (x_orig, x_res)) in result_original
            .x
            .iter()
            .zip(result_resident.x.iter())
            .enumerate()
        {
            assert!(
                (x_orig - x_res).abs() < 1e-8,
                "Solution mismatch at {}: orig={}, resident={}",
                i,
                x_orig,
                x_res
            );
        }
    }

    #[tokio::test]
    async fn test_cg_gpu_preconditioned() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return; // Skip if no f64 GPU
        };

        // Test preconditioned CG
        let a = create_spd_tridiagonal(100);
        let b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

        let result = CgGpu::solve_preconditioned(device.clone(), &a, &b, 1e-10, 500, 10).unwrap();

        assert!(result.converged, "Preconditioned CG should converge");
        assert!(
            result.residual < 1e-10,
            "Residual should be small: {}",
            result.residual
        );

        // Verify: Ax ≈ b
        let ax = a.matvec(&result.x).unwrap();
        for (i, (axi, bi)) in ax.iter().zip(b.iter()).enumerate() {
            assert!(
                (axi - bi).abs() < 1e-6,
                "Ax[{}] = {} should equal b[{}] = {}",
                i,
                axi,
                i,
                bi
            );
        }
    }

    #[tokio::test]
    async fn test_cg_preconditioned_vs_unpreconditioned() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return; // Skip if no f64 GPU
        };

        // Compare iteration counts: preconditioned should need fewer iterations
        let a = create_spd_tridiagonal(100);
        let b: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

        let result_unprecond =
            CgGpu::solve_gpu_resident(device.clone(), &a, &b, 1e-10, 500, 1).unwrap();
        let result_precond =
            CgGpu::solve_preconditioned(device.clone(), &a, &b, 1e-10, 500, 1).unwrap();

        assert!(
            result_unprecond.converged,
            "Unpreconditioned should converge"
        );
        assert!(result_precond.converged, "Preconditioned should converge");

        // For this specific matrix (tridiagonal with constant diagonal),
        // both should converge quickly, but let's at least verify they both work
        println!(
            "Iterations: unprecond={}, precond={}",
            result_unprecond.iterations, result_precond.iterations
        );

        // Solutions should be nearly identical
        for (i, (x_u, x_p)) in result_unprecond
            .x
            .iter()
            .zip(result_precond.x.iter())
            .enumerate()
        {
            assert!(
                (x_u - x_p).abs() < 1e-6,
                "Solution mismatch at {}: unprecond={}, precond={}",
                i,
                x_u,
                x_p
            );
        }
    }
}
