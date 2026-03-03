// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use std::sync::Arc;

fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
    crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
}

#[test]
fn test_coulomb_f64_two_particles() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = CoulombForceF64::new(device).unwrap();

    // Two particles with opposite charges
    let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // Along x-axis, 1 unit apart
    let charges = vec![1.0, -1.0]; // Opposite charges

    let forces = op
        .compute_forces(&positions, &charges, Some(1.0), None, Some(1e-10))
        .unwrap();

    // Force should attract: F ~ k * q1 * q2 / r^2 = 1 * 1 * (-1) / 1 = -1
    // Particle 0 should be pulled toward particle 1 (positive x direction)
    assert!(forces[0] > 0.0, "F_x on particle 0 should be positive");
    assert!(forces[1].abs() < 1e-10, "F_y on particle 0 should be ~0");
    assert!(forces[2].abs() < 1e-10, "F_z on particle 0 should be ~0");

    // Particle 1 should be pulled toward particle 0 (negative x direction)
    assert!(forces[3] < 0.0, "F_x on particle 1 should be negative");

    // Forces should be equal and opposite
    assert!(
        (forces[0] + forces[3]).abs() < 1e-10,
        "Forces should sum to zero"
    );
}

#[test]
fn test_coulomb_f64_repulsion() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = CoulombForceF64::new(device).unwrap();

    // Two particles with same charges
    let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let charges = vec![1.0, 1.0]; // Same sign

    let forces = op
        .compute_forces(&positions, &charges, Some(1.0), None, Some(1e-10))
        .unwrap();

    // Force should repel: particle 0 pushed in negative x direction
    assert!(
        forces[0] < 0.0,
        "F_x on particle 0 should be negative (repulsion)"
    );
    assert!(
        forces[3] > 0.0,
        "F_x on particle 1 should be positive (repulsion)"
    );
}

#[test]
fn test_coulomb_f64_distance_scaling() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = CoulombForceF64::new(device).unwrap();

    // Two particles at distance 1
    let positions1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let charges = vec![1.0, 1.0];

    let forces1 = op
        .compute_forces(&positions1, &charges, Some(1.0), None, Some(1e-10))
        .unwrap();

    // Two particles at distance 2
    let positions2 = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let forces2 = op
        .compute_forces(&positions2, &charges, Some(1.0), None, Some(1e-10))
        .unwrap();

    // Force should scale as 1/r^2, so F(2) = F(1)/4
    let ratio = forces1[0].abs() / forces2[0].abs();
    assert!(
        (ratio - 4.0).abs() < 0.01,
        "Force should scale as 1/r^2, ratio = {}",
        ratio
    );
}

#[test]
fn test_coulomb_f64_with_energy_gpu() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = CoulombForceF64::new(device).unwrap();

    // Need at least 32 particles to use GPU path
    let n = 40;
    let mut positions = vec![0.0; n * 3];
    let mut charges = vec![0.0; n];

    // Arrange particles in a line with alternating charges
    for i in 0..n {
        positions[i * 3] = i as f64; // x position
        charges[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let (forces, energies) = op
        .compute_forces_and_energy(&positions, &charges, Some(1.0), None, Some(1e-10))
        .unwrap();

    assert_eq!(forces.len(), n * 3, "Forces should have 3N elements");
    assert_eq!(energies.len(), n, "Energies should have N elements");

    // Total energy should be negative (attractive system with alternating charges)
    let total_energy: f64 = energies.iter().sum();
    assert!(
        total_energy < 0.0,
        "Total energy should be negative for alternating charges"
    );

    // Forces on interior particles should be small (nearly balanced)
    // First and last particles see unbalanced forces
    let mid = n / 2;
    let fx_mid = forces[mid * 3].abs();
    let fx_first = forces[0].abs();

    // Interior forces should be smaller than boundary forces
    assert!(
        fx_mid < fx_first,
        "Interior force {} should be less than boundary force {}",
        fx_mid,
        fx_first
    );
}
