// SPDX-License-Identifier: AGPL-3.0-or-later
//! BORN-MAYER F64 - Short-range repulsive force - f64 precision WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//!
//! Applications:
//! - Ionic crystals (NaCl, MgO)
//! - Hard-core repulsion in MD
//! - Steric effects modeling

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

const WGSL_DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
const WGSL_DF64_TRANSCENDENTALS: &str =
    include_str!("../../../shaders/math/df64_transcendentals.wgsl");
const BM_SHADER_DF64: &str = include_str!("born_mayer_df64.wgsl");

/// f64 Born-Mayer force calculator
///
/// Potential: U(r) = A * exp(-r/ρ)
/// Force: F = (A/ρ) * exp(-r/ρ) * r̂
///
/// GPU shader dispatch via `born_mayer_f64.wgsl` (N-body direct).
pub struct BornMayerForceF64 {
    device: Arc<WgpuDevice>,
}

impl BornMayerForceF64 {
    /// Create Born-Mayer force calculator.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("born_mayer_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let profile = GpuDriverProfile::from_device(device);
        let strategy = profile.fp64_strategy();
        tracing::info!(
            ?strategy,
            "BornMayer F64: using {:?} FP64 strategy",
            strategy
        );
        match strategy {
            Fp64Strategy::Native | Fp64Strategy::Concurrent => Self::wgsl_shader().to_string(),
            Fp64Strategy::Hybrid => {
                format!("{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{BM_SHADER_DF64}")
            }
        }
    }

    /// Compute Born-Mayer forces (always GPU dispatch).
    pub fn compute_forces(
        &self,
        positions: &[f64],
        a_params: &[f64],
        rho_params: &[f64],
        cutoff: f64,
    ) -> Result<Vec<f64>> {
        let n = positions.len() / 3;
        self.compute_gpu(positions, a_params, rho_params, cutoff, n)
    }

    /// Compute forces (GPU) and total potential energy.
    ///
    /// Forces come from GPU dispatch; energy is accumulated on the host from
    /// the same Born-Mayer potential U(r) = A·exp(−r/ρ) using geometric
    /// mixing rules: A = √(Aᵢ·Aⱼ), ρ = (ρᵢ+ρⱼ)/2.
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        a_params: &[f64],
        rho_params: &[f64],
        cutoff: f64,
    ) -> Result<(Vec<f64>, f64)> {
        let forces = self.compute_forces(positions, a_params, rho_params, cutoff)?;

        let n = positions.len() / 3;
        let cutoff_sq = cutoff * cutoff;
        let mut energy = 0.0_f64;
        for i in 0..n {
            let pi = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            for j in (i + 1)..n {
                let pj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
                let rv = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                let r_sq = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
                if r_sq > cutoff_sq || r_sq < 1e-20 {
                    continue;
                }
                let r = r_sq.sqrt();
                let a = (a_params[i] * a_params[j]).sqrt();
                let rho = (rho_params[i] + rho_params[j]) * 0.5;
                energy += a * (-r / rho).exp();
            }
        }

        Ok((forces, energy))
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        a_params: &[f64],
        rho_params: &[f64],
        cutoff: f64,
        n: usize,
    ) -> Result<Vec<f64>> {
        let dev = &self.device;

        let to_buf = |label: &str, data: &[f64], usage: wgpu::BufferUsages| -> wgpu::Buffer {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            dev.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &bytes,
                    usage,
                })
        };

        let pos_buf = to_buf("bm pos", positions, wgpu::BufferUsages::STORAGE);
        let a_buf = to_buf("bm A", a_params, wgpu::BufferUsages::STORAGE);
        let rho_buf = to_buf("bm rho", rho_params, wgpu::BufferUsages::STORAGE);

        let forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bm forces"),
            size: (n * 3 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            _pad0: u32,
            cutoff_lo: u32,
            cutoff_hi: u32,
        }

        let cutoff_bits = cutoff.to_bits();
        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    n_particles: n as u32,
                    _pad0: 0,
                    cutoff_lo: cutoff_bits as u32,
                    cutoff_hi: (cutoff_bits >> 32) as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "born_mayer_f64")
            .shader(&src, "main")
            .f64()
            .storage_read(0, &pos_buf)
            .storage_read(1, &a_buf)
            .storage_read(2, &rho_buf)
            .storage_rw(3, &forces_buf)
            .uniform(4, &params_buf)
            .dispatch(wg, 1, 1)
            .submit();

        dev.read_f64_buffer(&forces_buf, n * 3)
    }

    /// CPU reference (test/validation only).
    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn compute_cpu(
        &self,
        positions: &[f64],
        a_params: &[f64],
        rho_params: &[f64],
        cutoff: f64,
    ) -> Vec<f64> {
        let n = positions.len() / 3;
        let mut forces = vec![0.0; n * 3];
        let cutoff_sq = cutoff * cutoff;

        for i in 0..n {
            let pi = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            let ai = a_params[i];
            let rhoi = rho_params[i];

            for j in 0..n {
                if i == j {
                    continue;
                }

                let pj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
                let aj = a_params[j];
                let rhoj = rho_params[j];

                let r_vec = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                let r_sq = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

                if r_sq > cutoff_sq || r_sq < 1e-20 {
                    continue;
                }

                let r = r_sq.sqrt();

                // Geometric mixing rules
                let a = (ai * aj).sqrt();
                let rho = (rhoi + rhoj) * 0.5;

                // F = (A/ρ) * exp(-r/ρ) * r̂
                let exp_term = (-r / rho).exp();
                let force_mag = (a / rho) * exp_term;

                let inv_r = 1.0 / r;
                forces[i * 3] += force_mag * r_vec[0] * inv_r;
                forces[i * 3 + 1] += force_mag * r_vec[1] * inv_r;
                forces[i * 3 + 2] += force_mag * r_vec[2] * inv_r;
            }
        }

        forces
    }

    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn compute_cpu_with_energy(
        &self,
        positions: &[f64],
        a_params: &[f64],
        rho_params: &[f64],
        cutoff: f64,
    ) -> (Vec<f64>, f64) {
        let n = positions.len() / 3;
        let mut forces = vec![0.0; n * 3];
        let mut energy = 0.0;
        let cutoff_sq = cutoff * cutoff;

        for i in 0..n {
            let pi = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            let ai = a_params[i];
            let rhoi = rho_params[i];

            for j in (i + 1)..n {
                let pj = [positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]];
                let aj = a_params[j];
                let rhoj = rho_params[j];

                let r_vec = [pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]];
                let r_sq = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1] + r_vec[2] * r_vec[2];

                if r_sq > cutoff_sq || r_sq < 1e-20 {
                    continue;
                }

                let r = r_sq.sqrt();

                let a = (ai * aj).sqrt();
                let rho = (rhoi + rhoj) * 0.5;

                let exp_term = (-r / rho).exp();

                // U = A * exp(-r/ρ)
                energy += a * exp_term;

                // F = (A/ρ) * exp(-r/ρ) * r̂
                let force_mag = (a / rho) * exp_term;
                let inv_r = 1.0 / r;
                let f = [
                    force_mag * r_vec[0] * inv_r,
                    force_mag * r_vec[1] * inv_r,
                    force_mag * r_vec[2] * inv_r,
                ];

                // Newton's third law
                forces[i * 3] += f[0];
                forces[i * 3 + 1] += f[1];
                forces[i * 3 + 2] += f[2];
                forces[j * 3] -= f[0];
                forces[j * 3 + 1] -= f[1];
                forces[j * 3 + 2] -= f[2];
            }
        }

        (forces, energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_born_mayer_two_particles() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bm = BornMayerForceF64::new(device)?;

        // Two particles along x-axis
        let positions = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let a_params = vec![1.0, 1.0];
        let rho_params = vec![1.0, 1.0];

        let forces = bm.compute_forces(&positions, &a_params, &rho_params, 10.0)?;

        // Force should be repulsive (positive x for particle 0, negative for particle 1)
        assert!(forces[0] > 0.0, "Particle 0 should be pushed in -x");
        assert!(forces[3] < 0.0, "Particle 1 should be pushed in +x");
        // Newton's third law
        assert!(
            (forces[0] + forces[3]).abs() < 1e-10,
            "Forces should be equal and opposite"
        );
        Ok(())
    }

    #[test]
    fn test_born_mayer_energy_positive() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bm = BornMayerForceF64::new(device)?;

        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a_params = vec![1.0, 1.0];
        let rho_params = vec![0.5, 0.5];

        let (_, energy) = bm.compute_forces_and_energy(&positions, &a_params, &rho_params, 10.0)?;

        // Born-Mayer is purely repulsive, so energy > 0
        assert!(energy > 0.0, "Born-Mayer energy should be positive");
        Ok(())
    }

    #[test]
    fn test_born_mayer_cutoff() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let bm = BornMayerForceF64::new(device)?;

        let positions = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let a_params = vec![1.0, 1.0];
        let rho_params = vec![0.5, 0.5];

        // With cutoff = 3, particles at distance 5 should not interact
        let forces = bm.compute_forces(&positions, &a_params, &rho_params, 3.0)?;

        for f in forces {
            assert!(f.abs() < 1e-15, "No force expected beyond cutoff");
        }
        Ok(())
    }
}
