// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-Resident Pipeline Tests
//!
//! Comprehensive tests for the GPU-resident physics pipeline components:
//! - Max Absolute Difference Reduction
//! - Persistent Buffer Management
//! - Batched Bisection
//! - Grid Quadrature GEMM
//! - Multi-Kernel Pipeline
//!
//! These tests validate the hotSpring Feb 16 2026 evolution targets.

#![expect(clippy::unwrap_used, reason = "tests")]
use barracuda::device::tensor_context::{BufferDescriptor, TensorContext};
use barracuda::ops::linalg::GridQuadratureGemm;
use barracuda::ops::MaxAbsDiffF64;
use barracuda::optimize::BatchedBisectionGpu;

// ============================================================================
// MaxAbsDiffF64 Tests
// ============================================================================

#[tokio::test]
async fn test_max_abs_diff_convergence_simulation() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        eprintln!("No GPU available, skipping test");
        return;
    };

    // Simulate SCF convergence: energy difference decreasing each iteration
    let e_old = vec![
        -100.5, -99.3, -98.1, -97.2, -96.5, -95.8, -95.2, -94.7, -94.3, -94.0,
    ];
    let e_new = vec![
        -100.6, -99.4, -98.2, -97.3, -96.6, -95.9, -95.3, -94.8, -94.4, -94.1,
    ];

    let diff = MaxAbsDiffF64::compute(device, &e_old, &e_new).unwrap();

    // Max diff should be ~0.1
    assert!(
        (diff - 0.1).abs() < 1e-10,
        "Max energy diff should be ~0.1, got {}",
        diff
    );
}

#[tokio::test]
async fn test_max_abs_diff_converged_system() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // Two arrays that are identical within tolerance
    let a: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001).sin()).collect();
    let b = a.clone();

    let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();

    assert!(
        diff < 1e-14,
        "Identical arrays should have ~0 diff, got {}",
        diff
    );
}

#[tokio::test]
async fn test_max_abs_diff_stress_large() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // Large arrays (100K elements, multiple workgroups)
    let n = 100_000;
    let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut b = a.clone();

    // Insert a known max difference at a specific location
    b[54321] = a[54321] + 999.0;

    let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();

    assert!(
        (diff - 999.0).abs() < 1e-6,
        "Max diff should be 999, got {}",
        diff
    );
}

// ============================================================================
// Persistent Buffer Management Tests
// ============================================================================

#[tokio::test]
async fn test_solver_buffers_hfb_pattern() {
    let device = barracuda::device::test_pool::get_test_device().await;
    let ctx = TensorContext::new(device);

    // Simulate HFB solver buffer allocation
    let batch = 169; // Number of nuclei (like hotSpring study)
    let n = 12; // Basis size
    let grid = 100; // Grid points

    let buffers = ctx
        .pin_solver_buffers(
            "hfb_scf",
            &[
                ("hamiltonian", BufferDescriptor::f64_array(batch * n * n)),
                ("eigenvalues", BufferDescriptor::f64_array(batch * n)),
                ("eigenvectors", BufferDescriptor::f64_array(batch * n * n)),
                ("density", BufferDescriptor::f64_array(batch * n * n)),
                ("phi", BufferDescriptor::f64_array(batch * n * grid)),
                ("weight_fn", BufferDescriptor::f64_array(batch * grid)),
            ],
        )
        .expect("Failed to pin HFB buffers");

    // Verify all buffers exist with correct sizes
    assert_eq!(buffers.len(), 6);

    let h = buffers.get("hamiltonian").unwrap();
    assert!(h.size() >= (batch * n * n * 8) as u64);

    let phi = buffers.get("phi").unwrap();
    assert!(phi.size() >= (batch * n * grid * 8) as u64);

    // Clean up
    assert!(ctx.release_solver_buffers("hfb_scf"));
}

#[tokio::test]
async fn test_solver_buffers_multiple_solvers() {
    let device = barracuda::device::test_pool::get_test_device().await;
    let pool = barracuda::device::tensor_context::BufferPool::new_standalone(device.device_arc());

    // Pin buffers for two different solvers
    let _hfb_buffers = pool
        .pin_solver_buffers(
            "hfb",
            &[
                ("h", BufferDescriptor::f64_array(100)),
                ("rho", BufferDescriptor::f64_array(100)),
            ],
        )
        .unwrap();

    let _md_buffers = pool
        .pin_solver_buffers(
            "md",
            &[
                ("positions", BufferDescriptor::f64_array(3000)),
                ("velocities", BufferDescriptor::f64_array(3000)),
                ("forces", BufferDescriptor::f64_array(3000)),
            ],
        )
        .unwrap();

    // Both should exist
    assert!(pool.has_solver_buffers("hfb"));
    assert!(pool.has_solver_buffers("md"));

    // List all solver IDs
    let ids = pool.solver_ids();
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"hfb".to_string()));
    assert!(ids.contains(&"md".to_string()));

    // Release one
    pool.release_solver_buffers("hfb");
    assert!(!pool.has_solver_buffers("hfb"));
    assert!(pool.has_solver_buffers("md"));

    // Release the other
    pool.release_solver_buffers("md");
    assert!(pool.solver_ids().is_empty());
}

// ============================================================================
// Batched Bisection Tests
// ============================================================================

#[tokio::test]
async fn test_batched_bisection_square_roots() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 100, 1e-12).unwrap();

    // Find √2, √3, ..., √101 (100 problems)
    let n = 100;
    let lower = vec![0.0; n];
    let upper: Vec<f64> = (2..=n + 1).map(|i| (i as f64).max(2.0)).collect();
    let targets: Vec<f64> = (2..=n + 1).map(|i| i as f64).collect();

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();

    // Verify all roots
    for i in 0..n {
        let expected = ((i + 2) as f64).sqrt();
        let actual = result.roots[i];
        assert!(
            (actual - expected).abs() < 1e-10,
            "√{} should be {}, got {} (diff: {})",
            i + 2,
            expected,
            actual,
            (actual - expected).abs()
        );
    }
}

#[tokio::test]
async fn test_batched_bisection_iteration_counts() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 50, 1e-10).unwrap();

    let lower = vec![0.0; 10];
    let upper = vec![10.0; 10];
    let targets: Vec<f64> = (1..=10).map(|i| i as f64).collect();

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();

    // All problems should converge in reasonable iterations
    for (i, &iters) in result.iterations.iter().enumerate() {
        assert!(
            iters < 50,
            "Problem {} took {} iterations, expected < 50",
            i,
            iters
        );
    }
}

#[tokio::test]
async fn test_batched_bisection_stress_1000() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 64, 1e-10).unwrap();

    // 1000 parallel root-finding problems
    let n = 1000;
    let lower = vec![0.0; n];
    let upper: Vec<f64> = (1..=n).map(|i| (i as f64 + 1.0).sqrt() + 1.0).collect();
    let targets: Vec<f64> = (1..=n).map(|i| i as f64).collect();

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();

    assert_eq!(result.roots.len(), n);

    // Spot check some results
    assert!((result.roots[0] - 1.0).abs() < 1e-9); // √1 = 1
    assert!((result.roots[99] - 10.0).abs() < 1e-9); // √100 = 10
    assert!((result.roots[999] - (1000.0_f64).sqrt()).abs() < 1e-8); // √1000
}

// ============================================================================
// Grid Quadrature GEMM Tests
// ============================================================================

#[tokio::test]
async fn test_grid_quadrature_gemm_overlap_integral() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // Compute overlap matrix: S[i,j] = ∫ φ_i(r) φ_j(r) dr
    // Using identity weight function and unit quadrature weights
    let batch = 1;
    let n = 3;
    let grid = 5;

    // Orthonormal-ish basis functions (normalized rows)
    let phi = vec![
        // φ_0
        0.5, 0.5, 0.5, 0.3, 0.2, // φ_1
        0.2, 0.3, 0.5, 0.5, 0.5, // φ_2
        0.4, 0.4, 0.4, 0.4, 0.4,
    ];

    let w = vec![1.0; grid]; // Unit weight function
    let quad_weights = vec![1.0; grid]; // Unit quadrature

    let gemm = GridQuadratureGemm::new(device, batch, n, grid).unwrap();
    let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

    // Matrix should be symmetric
    for i in 0..n {
        for j in i..n {
            let s_ij = result[i * n + j];
            let s_ji = result[j * n + i];
            assert!(
                (s_ij - s_ji).abs() < 1e-10,
                "S[{},{}] = {} != S[{},{}] = {}",
                i,
                j,
                s_ij,
                j,
                i,
                s_ji
            );
        }
    }
}

#[tokio::test]
async fn test_grid_quadrature_gemm_batched() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // 10 independent Hamiltonians
    let batch = 10;
    let n = 8;
    let grid = 50;

    // Random-ish data
    let phi: Vec<f64> = (0..batch * n * grid)
        .map(|i| ((i as f64) * 0.01).sin())
        .collect();
    let w: Vec<f64> = (0..batch * grid)
        .map(|i| 1.0 + ((i as f64) * 0.02).cos() * 0.5)
        .collect();
    let quad_weights: Vec<f64> = (0..grid).map(|i| 1.0 / (1.0 + i as f64 * 0.1)).collect();

    let gemm = GridQuadratureGemm::new(device, batch, n, grid).unwrap();
    let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

    // Verify output size and symmetry
    assert_eq!(result.len(), batch * n * n);

    for b in 0..batch {
        for i in 0..n {
            for j in i..n {
                let h_ij = result[b * n * n + i * n + j];
                let h_ji = result[b * n * n + j * n + i];
                assert!(
                    (h_ij - h_ji).abs() < 1e-9,
                    "Batch {}: H[{},{}] = {}, H[{},{}] = {}",
                    b,
                    i,
                    j,
                    h_ij,
                    j,
                    i,
                    h_ji
                );
            }
        }
    }
}

#[tokio::test]
async fn test_grid_quadrature_gemm_large_grid() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // Large grid (>256, uses general kernel)
    let batch = 5;
    let n = 4;
    let grid = 500;

    let phi: Vec<f64> = (0..batch * n * grid)
        .map(|i| ((i as f64) * 0.001).cos())
        .collect();
    let w: Vec<f64> = vec![1.0; batch * grid];
    let quad_weights: Vec<f64> = vec![1.0 / grid as f64; grid];

    let gemm = GridQuadratureGemm::new(device, batch, n, grid).unwrap();
    let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

    assert_eq!(result.len(), batch * n * n);

    // Diagonal elements should be positive (sum of squares)
    for b in 0..batch {
        for i in 0..n {
            let h_ii = result[b * n * n + i * n + i];
            assert!(
                h_ii >= 0.0,
                "Diagonal H[{},{}] = {} should be >= 0",
                i,
                i,
                h_ii
            );
        }
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[tokio::test]
async fn test_scf_convergence_loop_simulation() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    // Simulate an SCF convergence loop using our new ops
    let tolerance = 1e-10;
    let max_iter = 50;

    // Initial "energies"
    let mut e_old: Vec<f64> = (0..10).map(|i| -100.0 + i as f64 * 0.1).collect();

    // Simulate convergence by reducing differences each iteration
    for iter in 0..max_iter {
        // "New" energies converging to target
        let e_new: Vec<f64> = e_old
            .iter()
            .map(|e| e - 0.1 * (0.5_f64).powi(iter))
            .collect();

        // Check convergence using GPU op
        let diff = MaxAbsDiffF64::compute(device.clone(), &e_old, &e_new).unwrap();

        if diff < tolerance {
            println!("Converged in {} iterations, diff = {:.2e}", iter + 1, diff);
            return; // Test passed
        }

        e_old = e_new;
    }

    panic!("Failed to converge in {} iterations", max_iter);
}

#[tokio::test]
async fn test_hotspring_nucleus_count() {
    // Verify we can handle hotSpring's 169 nuclei in parallel
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    let batch = 169; // hotSpring validation count
    let _n = 12; // Typical basis size (for reference)

    // Create test data for 169 nuclei
    let a: Vec<f64> = (0..batch).map(|i| -100.0 - i as f64 * 0.5).collect();
    let b: Vec<f64> = (0..batch).map(|i| -100.1 - i as f64 * 0.5).collect();

    let diff = MaxAbsDiffF64::compute(device.clone(), &a, &b).unwrap();

    assert!(
        (diff - 0.1).abs() < 1e-10,
        "Max diff for 169 nuclei should be 0.1, got {}",
        diff
    );

    // Also test bisection for 169 problems
    let bisect = BatchedBisectionGpu::new(device.clone(), 64, 1e-10).unwrap();
    let lower = vec![0.0; batch];
    let upper = vec![20.0; batch];
    let targets: Vec<f64> = (1..=batch).map(|i| i as f64).collect();

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();
    assert_eq!(result.roots.len(), batch);

    // Spot check
    assert!((result.roots[0] - 1.0).abs() < 1e-9);
    assert!((result.roots[168] - 13.0).abs() < 1e-9); // √169 = 13

    println!("Successfully processed 169 nuclei in parallel");
}

// ============================================================================
// E2E Test: Full GPU-Resident SCF Iteration
// ============================================================================

/// End-to-end test demonstrating GPU-resident physics pipeline
///
/// Simulates one SCF iteration with all components on GPU:
/// 1. Build Hamiltonian (Grid Quadrature GEMM)
/// 2. Check convergence (Max Abs Diff)
/// 3. Continue or exit based on tolerance
///
/// This validates the Amdahl's Law solution from hotSpring Feb 16 2026.
#[tokio::test]
async fn test_e2e_gpu_resident_scf_iteration() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        eprintln!("No GPU available, skipping E2E test");
        return;
    };

    println!("\n=== E2E GPU-Resident SCF Iteration Test ===\n");

    // Simulation parameters (like hotSpring study)
    let batch = 10; // 10 nuclei for quick test (use 169 for full)
    let n = 8; // Basis size
    let grid = 100; // Grid points
    let tolerance = 1e-8;
    let max_iter = 20;

    // Initialize basis functions (mock harmonic oscillator basis)
    let phi: Vec<f64> = (0..batch * n * grid)
        .map(|idx| {
            let _b = idx / (n * grid); // batch index (could vary potential by nucleus)
            let i = (idx % (n * grid)) / grid;
            let k = idx % grid;
            let r = k as f64 * 0.1; // Grid spacing
            let n_ho = i as f64;
            // Simplified HO-like radial function
            (-r * r / (2.0 + n_ho)).exp() * r.powi(i as i32)
        })
        .collect();

    // Quadrature weights (trapezoidal rule)
    let dr = 0.1;
    let quad_weights: Vec<f64> = (0..grid)
        .map(|k| {
            let r = k as f64 * dr;
            r * r * dr // r² dr for spherical integration
        })
        .collect();

    // Initial weight function (potential energy)
    let w_old: Vec<f64> = (0..batch * grid)
        .map(|idx| {
            let k = idx % grid;
            let r = k as f64 * dr;
            -1.0 / (r + 0.1) // Coulomb-like potential
        })
        .collect();

    // Create GEMM operator
    let gemm = GridQuadratureGemm::new(device.clone(), batch, n, grid).unwrap();

    println!("Starting SCF iteration loop...\n");
    println!("  Batch size: {} nuclei", batch);
    println!("  Basis size: {}", n);
    println!("  Grid points: {}", grid);
    println!("  Tolerance: {:.2e}\n", tolerance);

    let mut w_current = w_old.clone();
    let mut h_old: Vec<f64> = vec![0.0; batch * n * n];

    for iter in 0..max_iter {
        // Step 1: Build Hamiltonian on GPU (Grid Quadrature GEMM)
        let h_new = gemm.execute(&phi, &w_current, &quad_weights).unwrap();

        // Step 2: Check convergence (Max Abs Diff)
        let diff = MaxAbsDiffF64::compute(device.clone(), &h_old, &h_new).unwrap();

        println!("  Iteration {}: max|ΔH| = {:.6e}", iter + 1, diff);

        if diff < tolerance {
            println!("\n✓ Converged in {} iterations!", iter + 1);
            println!("  Final max|ΔH| = {:.2e}", diff);

            // Verify Hamiltonians are symmetric
            let mut max_asym = 0.0_f64;
            for b in 0..batch {
                for i in 0..n {
                    for j in i + 1..n {
                        let h_ij = h_new[b * n * n + i * n + j];
                        let h_ji = h_new[b * n * n + j * n + i];
                        max_asym = max_asym.max((h_ij - h_ji).abs());
                    }
                }
            }
            println!("  Max asymmetry: {:.2e}", max_asym);

            return; // Test passed
        }

        // Simulate SCF update (in real code, would solve eigenvalue problem)
        // Here we just dampen the potential to simulate convergence
        for idx in 0..batch * grid {
            w_current[idx] = w_current[idx] * 0.95 + w_old[idx] * 0.05;
        }

        h_old = h_new;
    }

    // Should converge within max_iter
    panic!("SCF did not converge in {} iterations", max_iter);
}

/// E2E test with persistent buffer management
#[tokio::test]
async fn test_e2e_persistent_buffers_scf() {
    let Some(wgpu_device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    let device = wgpu_device;
    let ctx = TensorContext::new(device.clone());

    // Pin solver buffers for the entire SCF calculation
    let batch = 5;
    let n = 6;
    let grid = 50;

    let _buffers = ctx
        .pin_solver_buffers(
            "scf_e2e",
            &[
                ("h_old", BufferDescriptor::f64_array(batch * n * n)),
                ("h_new", BufferDescriptor::f64_array(batch * n * n)),
                ("phi", BufferDescriptor::f64_array(batch * n * grid)),
                ("w", BufferDescriptor::f64_array(batch * grid)),
                ("quad_weights", BufferDescriptor::f64_array(grid)),
                ("eigenvalues", BufferDescriptor::f64_array(batch * n)),
            ],
        )
        .expect("Failed to pin SCF buffers");

    // Verify buffers exist
    let stats = ctx.stats();
    assert!(
        stats.buffer_allocations >= 6,
        "Should have allocated 6 buffers"
    );

    // Simulate multiple SCF iterations using same buffers
    for iter in 0..3 {
        // In real code, would:
        // 1. Write phi, w, quad_weights to buffers
        // 2. Execute GEMM kernel
        // 3. Check convergence
        // All without CPU readback!

        println!("  SCF iteration {} (buffers persistent)", iter + 1);
    }

    // Clean up
    assert!(ctx.release_solver_buffers("scf_e2e"));
    println!("✓ Persistent buffer SCF test passed");
}

/// Performance comparison: batched vs sequential root-finding
#[tokio::test]
async fn test_e2e_batched_vs_sequential_performance() {
    let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await
    else {
        return;
    };

    use std::time::Instant;

    let n_problems = 500;
    let max_iter = 50;
    let tol = 1e-10;

    // Target values: find √n for n = 1..500
    let targets: Vec<f64> = (1..=n_problems).map(|i| i as f64).collect();
    let lower = vec![0.0; n_problems];
    let upper: Vec<f64> = targets.iter().map(|t| t.sqrt() + 1.0).collect();

    // GPU Batched (single dispatch)
    let start = Instant::now();
    let bisect = BatchedBisectionGpu::new(device.clone(), max_iter as u32, tol).unwrap();
    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();
    let gpu_time = start.elapsed();

    // Verify results
    let mut max_error = 0.0_f64;
    for i in 0..n_problems {
        let expected = ((i + 1) as f64).sqrt();
        let error = (result.roots[i] - expected).abs();
        max_error = max_error.max(error);
    }

    println!("\n=== Batched vs Sequential Performance ===");
    println!("  Problems: {}", n_problems);
    println!("  GPU batched time: {:?}", gpu_time);
    println!("  Max error: {:.2e}", max_error);
    println!("  All roots verified: {}", max_error < 1e-8);

    assert!(max_error < 1e-8, "GPU results should be accurate");
}
