// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[cfg(test)]
#[allow(dead_code)]
impl MorseForceF64 {
    fn compute_cpu(&self, positions: &[f64], bonds: &[MorseBond]) -> Vec<f64> {
        let n_particles = positions.len() / 3;
        let mut forces = vec![0.0f64; n_particles * 3];

        for bond in bonds {
            let (fx, fy, fz) = self.compute_bond_force(positions, bond);
            forces[bond.i as usize * 3] += fx;
            forces[bond.i as usize * 3 + 1] += fy;
            forces[bond.i as usize * 3 + 2] += fz;
            forces[bond.j as usize * 3] -= fx;
            forces[bond.j as usize * 3 + 1] -= fy;
            forces[bond.j as usize * 3 + 2] -= fz;
        }

        forces
    }

    fn compute_cpu_with_energy(
        &self,
        positions: &[f64],
        bonds: &[MorseBond],
    ) -> (Vec<f64>, Vec<f64>) {
        let n_particles = positions.len() / 3;
        let mut forces = vec![0.0f64; n_particles * 3];
        let mut energies = Vec::with_capacity(bonds.len());

        for bond in bonds {
            let (fx, fy, fz, energy) = self.compute_bond_force_and_energy(positions, bond);
            forces[bond.i as usize * 3] += fx;
            forces[bond.i as usize * 3 + 1] += fy;
            forces[bond.i as usize * 3 + 2] += fz;
            forces[bond.j as usize * 3] -= fx;
            forces[bond.j as usize * 3 + 1] -= fy;
            forces[bond.j as usize * 3 + 2] -= fz;
            energies.push(energy);
        }

        (forces, energies)
    }

    fn compute_bond_force(&self, positions: &[f64], bond: &MorseBond) -> (f64, f64, f64) {
        let (dx, dy, dz, r) = bond_geometry(positions, bond);
        if r < 1e-10 {
            return (0.0, 0.0, 0.0);
        }

        let delta_r = r - bond.equilibrium_dist;
        let exp_term = (-bond.width_param * delta_r).exp();
        let one_minus_exp = 1.0 - exp_term;
        let force_magnitude =
            2.0 * bond.dissociation_energy * bond.width_param * one_minus_exp * exp_term;
        let force_over_r = force_magnitude / r;
        (force_over_r * dx, force_over_r * dy, force_over_r * dz)
    }

    fn compute_bond_force_and_energy(
        &self,
        positions: &[f64],
        bond: &MorseBond,
    ) -> (f64, f64, f64, f64) {
        let (dx, dy, dz, r) = bond_geometry(positions, bond);
        if r < 1e-10 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let delta_r = r - bond.equilibrium_dist;
        let exp_term = (-bond.width_param * delta_r).exp();
        let one_minus_exp = 1.0 - exp_term;
        let force_magnitude =
            2.0 * bond.dissociation_energy * bond.width_param * one_minus_exp * exp_term;
        let force_over_r = force_magnitude / r;
        let energy = bond.dissociation_energy * one_minus_exp * one_minus_exp;

        (
            force_over_r * dx,
            force_over_r * dy,
            force_over_r * dz,
            energy,
        )
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn bond_geometry(positions: &[f64], bond: &MorseBond) -> (f64, f64, f64, f64) {
    let dx = positions[bond.j as usize * 3] - positions[bond.i as usize * 3];
    let dy = positions[bond.j as usize * 3 + 1] - positions[bond.i as usize * 3 + 1];
    let dz = positions[bond.j as usize * 3 + 2] - positions[bond.i as usize * 3 + 2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    (dx, dy, dz, r)
}

fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
    crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
}

fn test_bond(r0: f64) -> MorseBond {
    MorseBond {
        i: 0,
        j: 1,
        dissociation_energy: 1.0,
        width_param: 2.0,
        equilibrium_dist: r0,
    }
}

#[test]
fn test_morse_equilibrium() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = MorseForceF64::new(device).unwrap();
    let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let bonds = vec![test_bond(1.0)];
    let forces = op.compute_forces(&positions, &bonds).unwrap();
    assert!(
        forces[0].abs() < 1e-10,
        "Force at equilibrium should be zero"
    );
    assert!(
        forces[3].abs() < 1e-10,
        "Force at equilibrium should be zero"
    );
}

#[test]
fn test_morse_stretched() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = MorseForceF64::new(device).unwrap();
    let positions = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
    let bonds = vec![test_bond(1.0)];
    let forces = op.compute_forces(&positions, &bonds).unwrap();
    assert!(
        forces[0] > 0.0,
        "Particle 0 should be pulled toward particle 1"
    );
    assert!(
        forces[3] < 0.0,
        "Particle 1 should be pulled toward particle 0"
    );
}

#[test]
fn test_morse_compressed() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = MorseForceF64::new(device).unwrap();
    let positions = vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
    let bonds = vec![test_bond(1.0)];
    let forces = op.compute_forces(&positions, &bonds).unwrap();
    assert!(
        forces[0] < 0.0,
        "Particle 0 should be pushed away from particle 1"
    );
    assert!(
        forces[3] > 0.0,
        "Particle 1 should be pushed away from particle 0"
    );
}

#[test]
fn test_morse_energy_minimum() {
    let Some(device) = get_test_device() else {
        return;
    };
    let op = MorseForceF64::new(device).unwrap();
    let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let bonds = vec![test_bond(1.0)];
    let (_, energies) = op.compute_forces_and_energy(&positions, &bonds).unwrap();
    assert!(
        energies[0].abs() < 1e-10,
        "Energy at equilibrium should be zero"
    );
}
