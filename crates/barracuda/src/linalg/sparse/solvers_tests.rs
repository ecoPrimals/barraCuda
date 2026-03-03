// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

fn create_spd_tridiagonal(n: usize) -> CsrMatrix {
    let mut triplets = Vec::new();

    for i in 0..n {
        triplets.push((i, i, 4.0)); // Main diagonal
        if i > 0 {
            triplets.push((i, i - 1, -1.0)); // Lower diagonal
        }
        if i < n - 1 {
            triplets.push((i, i + 1, -1.0)); // Upper diagonal
        }
    }

    CsrMatrix::from_triplets(n, n, &triplets)
}

#[test]
fn test_cg_small() {
    let a = create_spd_tridiagonal(3);
    let b = vec![1.0, 2.0, 3.0];

    let result = cg_solve(&a, &b, 1e-10, 100).unwrap();

    assert!(result.converged);
    assert!(result.residual < 1e-10);

    // Verify: Ax = b
    let ax = a.matvec(&result.x).unwrap();
    for (i, (&axi, &bi)) in ax.iter().zip(b.iter()).enumerate() {
        assert!(
            (axi - bi).abs() < 1e-8,
            "Mismatch at {}: {} vs {}",
            i,
            axi,
            bi
        );
    }
}

#[test]
fn test_cg_larger() {
    let a = create_spd_tridiagonal(100);
    let b: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();

    let result = cg_solve(&a, &b, 1e-10, 500).unwrap();

    assert!(result.converged);
    assert!(
        result.iterations < 200,
        "Too many iterations: {}",
        result.iterations
    );
}

#[test]
fn test_bicgstab_non_symmetric() {
    // Non-symmetric matrix
    let a = CsrMatrix::from_triplets(
        3,
        3,
        &[
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, -1.0),
            (1, 1, 3.0),
            (1, 2, 1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ],
    );

    let b = vec![1.0, 2.0, 3.0];
    let result = bicgstab_solve(&a, &b, 1e-10, 100).unwrap();

    assert!(result.converged);

    // Verify
    let ax = a.matvec(&result.x).unwrap();
    for (&axi, &bi) in ax.iter().zip(b.iter()) {
        assert!((axi - bi).abs() < 1e-8);
    }
}

#[test]
fn test_jacobi_diagonally_dominant() {
    // Diagonally dominant matrix
    let a = CsrMatrix::from_triplets(
        3,
        3,
        &[
            (0, 0, 5.0),
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 0, 1.0),
            (1, 1, 5.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 2, 5.0),
        ],
    );

    let b = vec![7.0, 7.0, 7.0];
    let result = jacobi_solve(&a, &b, 1e-8, 100).unwrap();

    assert!(result.converged);

    // Solution should be approximately [1, 1, 1]
    for &xi in &result.x {
        assert!((xi - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_solver_config() {
    let config = SolverConfig::default()
        .with_tolerance(1e-12)
        .with_max_iterations(500)
        .no_preconditioner();

    assert!((config.tolerance - 1e-12).abs() < 1e-14);
    assert_eq!(config.max_iterations, 500);
    assert!(!config.use_preconditioner);
}

#[test]
fn test_zero_rhs() {
    let a = create_spd_tridiagonal(3);
    let b = vec![0.0, 0.0, 0.0];

    let result = cg_solve(&a, &b, 1e-10, 100).unwrap();

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    for &xi in &result.x {
        assert!(xi.abs() < 1e-14);
    }
}
