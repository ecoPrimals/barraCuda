// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn test_ls_factor_computation() {
    // p1/2: l=1, j=1/2 -> ls = (0.75 - 2 - 0.75)/2 = -1
    let ls_p1_2 = compute_ls_factor(1, 0.5);
    assert!((ls_p1_2 - (-1.0)).abs() < 1e-10);

    // p3/2: l=1, j=3/2 -> ls = (3.75 - 2 - 0.75)/2 = 0.5
    let ls_p3_2 = compute_ls_factor(1, 1.5);
    assert!((ls_p3_2 - 0.5).abs() < 1e-10);

    // d3/2: l=2, j=3/2 -> ls = (3.75 - 6 - 0.75)/2 = -1.5
    let ls_d3_2 = compute_ls_factor(2, 1.5);
    assert!((ls_d3_2 - (-1.5)).abs() < 1e-10);

    // d5/2: l=2, j=5/2 -> ls = (8.75 - 6 - 0.75)/2 = 1.0
    let ls_d5_2 = compute_ls_factor(2, 2.5);
    assert!((ls_d5_2 - 1.0).abs() < 1e-10);

    // s1/2: l=0, j=1/2 -> ls = (0.75 - 0 - 0.75)/2 = 0
    let ls_s1_2 = compute_ls_factor(0, 0.5);
    assert!(ls_s1_2.abs() < 1e-10);
}

#[tokio::test]
#[ignore = "requires GPU hardware"]
async fn test_spin_orbit_basic() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let so = SpinOrbitGpu::new(device);

    // Simple test: 1 nucleus, 2 states, 5 grid points
    let n_grid = 5;
    let n_states = 2;
    let batch_size = 1;
    let dr = 0.5;
    let w0 = 120.0; // Typical Skyrme w0 in MeV·fm⁵

    // r_grid from dr to 5*dr
    let r_grid: Vec<f64> = (0..n_grid).map(|i| (i + 1) as f64 * dr).collect();

    // Simple wavefunction squared (normalized approximately)
    let mut wf_squared = vec![0.0; batch_size * n_states * n_grid];
    for i in 0..n_grid {
        let r = r_grid[i];
        // State 0: peaked at r=1
        wf_squared[i] = (-((r - 1.0).powi(2))).exp();
        // State 1: peaked at r=1.5
        wf_squared[n_grid + i] = (-((r - 1.5).powi(2))).exp();
    }

    // Density gradient (increasing)
    let drho_dr: Vec<f64> = (0..n_grid).map(|i| 0.1 * (i as f64 + 1.0)).collect();

    // ls factors
    let ls_factors = vec![0.5, -1.0]; // p3/2, p1/2 like

    let h_so = so
        .compute(&wf_squared, &drho_dr, &r_grid, &ls_factors, dr, w0)
        .unwrap();

    assert_eq!(h_so.len(), 2);

    // State 0 has positive ls, state 1 has negative ls
    // With positive drho/dr and w0, expect opposite signs
    assert!(
        h_so[0] * h_so[1] < 0.0 || h_so[0].abs() < 1e-10 || h_so[1].abs() < 1e-10,
        "States should have opposite-sign spin-orbit corrections"
    );
}

#[tokio::test]
#[ignore = "requires GPU hardware"]
async fn test_spin_orbit_with_density() {
    let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    else {
        return;
    };

    let so = SpinOrbitGpu::new(device);

    // Test the version that computes gradient internally
    let n_grid = 10;
    let dr = 0.5;
    let w0 = 120.0;

    let r_grid: Vec<f64> = (0..n_grid).map(|i| (i + 1) as f64 * dr).collect();

    // Linear density: ρ = 0.1 * r, so dρ/dr = 0.1
    let density: Vec<f64> = r_grid.iter().map(|r| 0.1 * r).collect();

    // Simple wavefunction
    let wf_squared: Vec<f64> = r_grid
        .iter()
        .map(|r| (-((r - 2.5).powi(2))).exp())
        .collect();

    let ls_factors = vec![1.0];

    let h_so = so
        .compute_with_density(&wf_squared, &density, &r_grid, &ls_factors, dr, w0)
        .unwrap();

    assert_eq!(h_so.len(), 1);
    assert!(
        h_so[0].abs() > 0.0,
        "Should have non-zero spin-orbit correction"
    );
}
