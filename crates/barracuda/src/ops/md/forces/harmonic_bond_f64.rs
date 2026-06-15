// SPDX-License-Identifier: AGPL-3.0-or-later
//! Harmonic Bond Force (f64) — Standard Bonded Interaction
//!
//! **Potential**: U(r) = ½k(r - r₀)²
//! **Force**: F = -k(r - r₀)·r̂
//!
//! Used by GROMOS 45a4, GLYCAM06, CHARMM36, AMBER, OPLS for covalent bonds.
//! Two-pass GPU dispatch following the Morse f64 pattern.

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy, WORKGROUP_SIZE_1D};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 harmonic bond force calculator (GPU-accelerated).
pub struct HarmonicBondF64 {
    device: Arc<WgpuDevice>,
}

/// Parameters for a single harmonic bond.
#[derive(Clone, Copy, Debug)]
pub struct HarmonicBond {
    /// Particle index i
    pub i: u32,
    /// Particle index j
    pub j: u32,
    /// Force constant k (kJ/mol/nm² or kcal/mol/Å²)
    pub force_constant: f64,
    /// Equilibrium bond length r₀ (nm or Å)
    pub eq_length: f64,
}

struct HarmonicBondBuffers {
    pos: wgpu::Buffer,
    pairs: wgpu::Buffer,
    force_constants: wgpu::Buffer,
    eq_lengths: wgpu::Buffer,
    bond_forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_bonds: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

impl HarmonicBondBuffers {
    fn new(dev: &WgpuDevice, positions: &[f64], bonds: &[HarmonicBond]) -> Self {
        let n_bonds = bonds.len();

        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hbond pos"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut pair_data = Vec::with_capacity(n_bonds * 2);
        let mut k_data = Vec::with_capacity(n_bonds);
        let mut r0_data = Vec::with_capacity(n_bonds);
        for b in bonds {
            pair_data.push(b.i);
            pair_data.push(b.j);
            k_data.push(b.force_constant);
            r0_data.push(b.eq_length);
        }

        let pairs = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("hbond pairs"),
                contents: bytemuck::cast_slice(&pair_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let to_f64_buf = |label: &str, data: &[f64]| -> wgpu::Buffer {
            dev.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let force_constants = to_f64_buf("hbond k", &k_data);
        let eq_lengths = to_f64_buf("hbond r0", &r0_data);

        let bond_forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hbond bf"),
            size: (n_bonds * 6 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    n_bonds: n_bonds as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        Self {
            pos,
            pairs,
            force_constants,
            eq_lengths,
            bond_forces,
            params,
        }
    }
}

fn reduce_bond_forces(
    dev: &WgpuDevice,
    shader_source: &str,
    bond_forces_buf: &wgpu::Buffer,
    pairs_buf: &wgpu::Buffer,
    n_particles: usize,
    n_bonds: usize,
) -> Result<Vec<f64>> {
    let particle_forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hbond pf"),
        size: (n_particles * 3 * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
    enc.clear_buffer(&particle_forces_buf, 0, None);
    dev.submit_commands(Some(enc.finish()));

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceParams {
        n_particles: u32,
        n_bonds: u32,
        _p0: u32,
        _p1: u32,
    }
    let rp_buf = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&ReduceParams {
                n_particles: n_particles as u32,
                n_bonds: n_bonds as u32,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let wg = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
    ComputeDispatch::new(dev, "reduce_hbond_forces_f64")
        .shader(shader_source, "reduce_bond_forces_f64")
        .f64()
        .uniform(0, &rp_buf)
        .storage_read(1, bond_forces_buf)
        .storage_read(2, pairs_buf)
        .storage_rw(3, &particle_forces_buf)
        .dispatch(wg, 1, 1)
        .submit()?;

    dev.read_f64_buffer(&particle_forces_buf, n_particles * 3)
}

impl HarmonicBondF64 {
    /// Create harmonic bond force calculator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("harmonic_bond_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let caps = DeviceCapabilities::from_device(device);
        match caps.fp64_strategy() {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => {
                // Harmonic bond uses only sqrt — no exp/log needed
                Self::wgsl_shader().to_string()
            }
        }
    }

    /// Compute harmonic bond forces for all bonds (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces(&self, positions: &[f64], bonds: &[HarmonicBond]) -> Result<Vec<f64>> {
        let n_particles = positions.len() / 3;
        if bonds.is_empty() {
            return Ok(vec![0.0f64; n_particles * 3]);
        }
        self.compute_gpu(positions, bonds, n_particles)
    }

    /// Compute harmonic bond forces and per-bond energies (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        bonds: &[HarmonicBond],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_particles = positions.len() / 3;
        if bonds.is_empty() {
            return Ok((vec![0.0f64; n_particles * 3], vec![]));
        }
        self.compute_gpu_with_energy(positions, bonds, n_particles)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        bonds: &[HarmonicBond],
        n_particles: usize,
    ) -> Result<Vec<f64>> {
        let n_bonds = bonds.len();
        let dev = &self.device;
        let buffers = HarmonicBondBuffers::new(dev, positions, bonds);

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_bonds as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "harmonic_bond_f64")
            .shader(&src, "harmonic_bond_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.pairs)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_lengths)
            .storage_rw(4, &buffers.bond_forces)
            .uniform(5, &buffers.params)
            .dispatch(wg, 1, 1)
            .submit()?;

        reduce_bond_forces(
            dev,
            &src,
            &buffers.bond_forces,
            &buffers.pairs,
            n_particles,
            n_bonds,
        )
    }

    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        bonds: &[HarmonicBond],
        n_particles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_bonds = bonds.len();
        let dev = &self.device;
        let buffers = HarmonicBondBuffers::new(dev, positions, bonds);

        let bond_energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hbond be"),
            size: (n_bonds * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_bonds as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "harmonic_bond_with_energy_f64")
            .shader(&src, "harmonic_bond_with_energy_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.pairs)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_lengths)
            .storage_rw(4, &buffers.bond_forces)
            .uniform(5, &buffers.params)
            .storage_rw(6, &bond_energy_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        let energies = dev.read_f64_buffer(&bond_energy_buf, n_bonds)?;
        let forces = reduce_bond_forces(
            dev,
            &src,
            &buffers.bond_forces,
            &buffers.pairs,
            n_particles,
            n_bonds,
        )?;
        Ok((forces, energies))
    }
}

#[cfg(test)]
#[expect(
    clippy::suboptimal_flops,
    reason = "reference physics math in textbook notation for GPU kernel verification"
)]
mod tests {
    use super::*;

    fn compute_cpu(positions: &[f64], bonds: &[HarmonicBond]) -> Vec<f64> {
        let n_particles = positions.len() / 3;
        let mut forces = vec![0.0f64; n_particles * 3];
        for b in bonds {
            let (i, j) = (b.i as usize, b.j as usize);
            let dx = positions[j * 3] - positions[i * 3];
            let dy = positions[j * 3 + 1] - positions[i * 3 + 1];
            let dz = positions[j * 3 + 2] - positions[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-10 {
                continue;
            }
            let f_mag = b.force_constant * (r - b.eq_length);
            let f_over_r = f_mag / r;
            forces[i * 3] += f_over_r * dx;
            forces[i * 3 + 1] += f_over_r * dy;
            forces[i * 3 + 2] += f_over_r * dz;
            forces[j * 3] -= f_over_r * dx;
            forces[j * 3 + 1] -= f_over_r * dy;
            forces[j * 3 + 2] -= f_over_r * dz;
        }
        forces
    }

    #[test]
    fn test_harmonic_bond_equilibrium_cpu() {
        let positions = vec![0.0, 0.0, 0.0, 0.15, 0.0, 0.0];
        let bonds = vec![HarmonicBond {
            i: 0,
            j: 1,
            force_constant: 250000.0,
            eq_length: 0.15,
        }];
        let forces = compute_cpu(&positions, &bonds);
        for f in &forces {
            assert!(
                f.abs() < 1e-10,
                "force at equilibrium should be zero, got {f}"
            );
        }
    }

    #[test]
    fn test_harmonic_bond_stretched_cpu() {
        let positions = vec![0.0, 0.0, 0.0, 0.20, 0.0, 0.0];
        let bonds = vec![HarmonicBond {
            i: 0,
            j: 1,
            force_constant: 250000.0,
            eq_length: 0.15,
        }];
        let forces = compute_cpu(&positions, &bonds);
        assert!(forces[0] > 0.0, "atom 0 should be pulled toward atom 1");
        assert!(forces[3] < 0.0, "atom 1 should be pulled toward atom 0");
        let expected_mag = 250000.0 * 0.05;
        assert!(
            (forces[0] - expected_mag).abs() < 1e-6,
            "force magnitude mismatch: {} vs {}",
            forces[0],
            expected_mag
        );
    }

    #[test]
    fn test_harmonic_bond_newton_third_law_cpu() {
        let positions = vec![0.0, 0.0, 0.0, 0.2, 0.1, 0.05];
        let bonds = vec![HarmonicBond {
            i: 0,
            j: 1,
            force_constant: 300000.0,
            eq_length: 0.14,
        }];
        let forces = compute_cpu(&positions, &bonds);
        for d in 0..3 {
            assert!(
                (forces[d] + forces[3 + d]).abs() < 1e-10,
                "Newton's 3rd law violated on dim {d}"
            );
        }
    }
}
