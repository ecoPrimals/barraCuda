// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[tokio::test]
async fn test_batched_bisection_polynomial() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 64, 1e-10).unwrap();

    // Find √2, √3, √5 in parallel
    let lower = vec![0.0, 0.0, 0.0];
    let upper = vec![2.0, 2.0, 3.0];
    let targets = vec![2.0, 3.0, 5.0];

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();

    assert!(
        (result.roots[0] - 2.0_f64.sqrt()).abs() < 1e-9,
        "√2 should be {}, got {}",
        2.0_f64.sqrt(),
        result.roots[0]
    );
    assert!(
        (result.roots[1] - 3.0_f64.sqrt()).abs() < 1e-9,
        "√3 should be {}, got {}",
        3.0_f64.sqrt(),
        result.roots[1]
    );
    assert!(
        (result.roots[2] - 5.0_f64.sqrt()).abs() < 1e-9,
        "√5 should be {}, got {}",
        5.0_f64.sqrt(),
        result.roots[2]
    );
}

#[tokio::test]
async fn test_batched_bisection_large_batch() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 64, 1e-10).unwrap();

    // Find √n for n = 2 to 1001 (1000 problems)
    let n = 1000;
    let lower = vec![0.0; n];
    let upper: Vec<f64> = (2..=n + 1).map(|i| (i as f64).max(2.0)).collect();
    let targets: Vec<f64> = (2..=n + 1).map(|i| i as f64).collect();

    let result = bisect.solve_polynomial(&lower, &upper, &targets).unwrap();

    // Verify all roots
    for i in 0..n {
        let expected = ((i + 2) as f64).sqrt();
        assert!(
            (result.roots[i] - expected).abs() < 1e-8,
            "Problem {}: √{} should be {}, got {}",
            i,
            i + 2,
            expected,
            result.roots[i]
        );
    }
}

#[tokio::test]
async fn test_batched_bisection_empty() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 64, 1e-10).unwrap();

    let result = bisect.solve_polynomial(&[], &[], &[]).unwrap();
    assert!(result.roots.is_empty());
    assert!(result.iterations.is_empty());
}

#[tokio::test]
async fn test_bcs_with_degeneracy() {
    // Test BCS with level degeneracies (hotSpring TIER 3.1)
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 100, 1e-10).unwrap();

    // Two problems:
    // Problem 0: 3 levels with degeneracies [2, 4, 2], ε = [0.0, 1.0, 2.0], Δ=0.5, N=4.0
    // Problem 1: 3 levels with degeneracies [1, 2, 1], ε = [0.0, 1.0, 2.0], Δ=0.5, N=2.0
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];

    // Packed eigenvalues [batch × n_levels]
    let eigenvalues = vec![
        0.0, 1.0, 2.0, // Problem 0
        0.0, 1.0, 2.0, // Problem 1
    ];

    // Degeneracies [batch × n_levels]
    let degeneracies = vec![
        2.0, 4.0, 2.0, // Problem 0: total capacity = 8
        1.0, 2.0, 1.0, // Problem 1: total capacity = 4
    ];

    let delta = vec![0.5, 0.5];
    let target_n = vec![4.0, 2.0];

    let result = bisect
        .solve_bcs_with_degeneracy(
            &lower,
            &upper,
            &eigenvalues,
            &degeneracies,
            &delta,
            &target_n,
        )
        .unwrap();

    assert_eq!(result.roots.len(), 2);

    // Verify: compute occupation for each root
    for (i, &mu) in result.roots.iter().enumerate() {
        let base = i * 3;
        let delta_i = delta[i];
        let target = target_n[i];

        let mut sum = 0.0;
        for k in 0..3 {
            let eps_k = eigenvalues[base + k];
            let deg_k = degeneracies[base + k];
            let diff = eps_k - mu;
            let e_k = (diff * diff + delta_i * delta_i).sqrt();
            let v2_k = 0.5 * (1.0 - diff / e_k);
            sum += deg_k * v2_k;
        }

        assert!(
            (sum - target).abs() < 0.01,
            "Problem {i}: Σ deg_k v²_k should be {target}, got {sum} (μ={mu})"
        );
    }
}

#[tokio::test]
async fn test_bcs_degeneracy_vs_no_degeneracy() {
    // Verify that uniform degeneracy=1 matches solve_bcs behavior
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let bisect = BatchedBisectionGpu::new(device, 100, 1e-10).unwrap();

    let lower = vec![-3.0];
    let upper = vec![5.0];
    let eigenvalues = vec![0.0, 1.0, 2.0, 3.0];
    let delta = vec![0.5];
    let target_n = vec![2.0];

    // Without degeneracy
    let result_no_deg = bisect
        .solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n)
        .unwrap();

    // With uniform degeneracy = 1.0
    let degeneracies = vec![1.0, 1.0, 1.0, 1.0];
    let result_with_deg = bisect
        .solve_bcs_with_degeneracy(
            &lower,
            &upper,
            &eigenvalues,
            &degeneracies,
            &delta,
            &target_n,
        )
        .unwrap();

    // Should match
    assert!(
        (result_no_deg.roots[0] - result_with_deg.roots[0]).abs() < 1e-6,
        "Results should match: {} vs {}",
        result_no_deg.roots[0],
        result_with_deg.roots[0]
    );
}
