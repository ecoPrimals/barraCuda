// SPDX-License-Identifier: AGPL-3.0-or-later

//! Stress virial GPU op — instantaneous stress tensor via ComputeDispatch.
//!
//! Computes the 6 independent components [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
//! from positions, velocities, forces, and masses using the virial theorem.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER_STRESS_VIRIAL: &str = include_str!("../../shaders/md/stress_virial_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StressVirialParams {
    n_atoms: u32,
    _pad0: u32,
    volume: f64,
}

/// Compute the 6-component stress tensor from particle data.
///
/// Returns `[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`.
///
/// # Arguments
/// * `positions` — `[N×3]` f64 particle positions
/// * `velocities` — `[N×3]` f64 particle velocities
/// * `forces` — `[N×3]` f64 total forces on each particle
/// * `masses` — `[N]` f64 particle masses
/// * `volume` — simulation box volume
pub fn compute_stress_virial(
    device: &Arc<WgpuDevice>,
    positions: &[f64],
    velocities: &[f64],
    forces: &[f64],
    masses: &[f64],
    volume: f64,
) -> Result<[f64; 6]> {
    let n_atoms = masses.len();
    assert_eq!(positions.len(), n_atoms * 3, "positions must be N×3");
    assert_eq!(velocities.len(), n_atoms * 3, "velocities must be N×3");
    assert_eq!(forces.len(), n_atoms * 3, "forces must be N×3");

    let pos_buf = device.create_buffer_f64_init("stress_virial:pos", positions);
    let vel_buf = device.create_buffer_f64_init("stress_virial:vel", velocities);
    let force_buf = device.create_buffer_f64_init("stress_virial:force", forces);
    let mass_buf = device.create_buffer_f64_init("stress_virial:mass", masses);
    let out_buf = device.create_buffer_f64(6)?;

    let params = StressVirialParams {
        n_atoms: n_atoms as u32,
        _pad0: 0,
        volume,
    };
    let params_buf = device.create_uniform_buffer("stress_virial:params", &params);

    ComputeDispatch::new(device, "stress_virial")
        .shader(SHADER_STRESS_VIRIAL, "main")
        .f64()
        .storage_read(0, &pos_buf)
        .storage_read(1, &vel_buf)
        .storage_read(2, &force_buf)
        .storage_read(3, &mass_buf)
        .storage_rw(4, &out_buf)
        .uniform(5, &params_buf)
        .dispatch(1, 1, 1)
        .submit();

    let result = device.read_f64_buffer(&out_buf, 6)?;
    Ok([
        result[0], result[1], result[2], result[3], result[4], result[5],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_virial_params_layout() {
        // repr(C): n_atoms(u32) + _pad0(u32) + volume(f64) = 8 + 8 = 16 bytes
        let params = StressVirialParams {
            n_atoms: 1000,
            _pad0: 0,
            volume: 1e-27,
        };
        assert_eq!(std::mem::size_of::<StressVirialParams>(), 16);
        assert_eq!(std::mem::align_of::<StressVirialParams>(), 8);
        assert_eq!(params.n_atoms, 1000);
    }

    #[test]
    fn test_stress_virial_shader_valid() {
        assert!(!SHADER_STRESS_VIRIAL.is_empty(), "shader must not be empty");
        assert!(
            SHADER_STRESS_VIRIAL.contains("fn main"),
            "shader must contain 'fn main'"
        );
    }
}
