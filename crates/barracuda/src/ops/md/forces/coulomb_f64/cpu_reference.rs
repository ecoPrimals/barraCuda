// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU reference implementations for Coulomb force (test/validation only).
//!
//! Production always dispatches to GPU shaders. These implementations
//! are used for unit test validation against the GPU output.

/// CPU reference for Coulomb forces (test/validation only).
#[cfg(test)]
#[allow(dead_code)]
pub fn compute_cpu(positions: &[f64], charges: &[f64], k: f64, cutoff: f64, eps: f64) -> Vec<f64> {
    let n = charges.len();
    let cutoff_sq = cutoff * cutoff;
    let eps_sq = eps * eps;
    let mut forces = vec![0.0f64; n * 3];

    for i in 0..n {
        let xi = positions[i * 3];
        let yi = positions[i * 3 + 1];
        let zi = positions[i * 3 + 2];
        let qi = charges[i];

        for j in 0..n {
            if i == j {
                continue;
            }

            let xj = positions[j * 3];
            let yj = positions[j * 3 + 1];
            let zj = positions[j * 3 + 2];
            let qj = charges[j];

            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;

            let r_sq = dx * dx + dy * dy + dz * dz + eps_sq;
            if r_sq > cutoff_sq {
                continue;
            }

            let r = r_sq.sqrt();
            let force_magnitude = k * qi * qj / r_sq;
            let force_over_r = -force_magnitude / r;

            forces[i * 3] += force_over_r * dx;
            forces[i * 3 + 1] += force_over_r * dy;
            forces[i * 3 + 2] += force_over_r * dz;
        }
    }

    forces
}

/// CPU reference for Coulomb forces with potential energy (test/validation only).
#[cfg(test)]
#[allow(dead_code)]
pub fn compute_cpu_with_energy(
    positions: &[f64],
    charges: &[f64],
    k: f64,
    cutoff: f64,
    eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = charges.len();
    let cutoff_sq = cutoff * cutoff;
    let eps_sq = eps * eps;
    let mut forces = vec![0.0f64; n * 3];
    let mut energies = vec![0.0f64; n];

    for i in 0..n {
        let xi = positions[i * 3];
        let yi = positions[i * 3 + 1];
        let zi = positions[i * 3 + 2];
        let qi = charges[i];

        for j in 0..n {
            if i == j {
                continue;
            }

            let xj = positions[j * 3];
            let yj = positions[j * 3 + 1];
            let zj = positions[j * 3 + 2];
            let qj = charges[j];

            let dx = xj - xi;
            let dy = yj - yi;
            let dz = zj - zi;

            let r_sq = dx * dx + dy * dy + dz * dz + eps_sq;
            if r_sq > cutoff_sq {
                continue;
            }

            let r = r_sq.sqrt();
            let force_magnitude = k * qi * qj / r_sq;
            let force_over_r = -force_magnitude / r;

            forces[i * 3] += force_over_r * dx;
            forces[i * 3 + 1] += force_over_r * dy;
            forces[i * 3 + 2] += force_over_r * dz;

            // Half to avoid double counting
            energies[i] += 0.5 * k * qi * qj / r;
        }
    }

    (forces, energies)
}
