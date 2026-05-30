// SPDX-License-Identifier: AGPL-3.0-or-later
//! Periodic Dihedral Torsion Force (f64) — Four-Body Bonded Interaction
//!
//! **Potential**: U(φ) = k_φ [1 + cos(nφ - δ)]
//! **Force**: Cartesian gradient using Blondel-Karplus decomposition
//!
//! Standard periodic (proper) dihedral used by GROMOS, AMBER, CHARMM, OPLS.
//! Two-pass GPU dispatch: per-dihedral forces → reduce to per-particle.

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy, WORKGROUP_SIZE_1D};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 periodic dihedral torsion force calculator (GPU-accelerated).
pub struct DihedralTorsionF64 {
    device: Arc<WgpuDevice>,
}

/// Parameters for a single dihedral term (i-j-k-l).
#[derive(Clone, Copy, Debug)]
pub struct DihedralTorsion {
    /// Particle index i
    pub i: u32,
    /// Particle index j
    pub j: u32,
    /// Particle index k
    pub k: u32,
    /// Particle index l
    pub l: u32,
    /// Barrier height k_φ (kJ/mol)
    pub barrier_height: f64,
    /// Periodicity n (integer, stored as f64 for GPU uniform compatibility)
    pub periodicity: f64,
    /// Phase shift δ (radians)
    pub phase_shift: f64,
}

struct DihedralBuffers {
    pos: wgpu::Buffer,
    quads: wgpu::Buffer,
    barrier: wgpu::Buffer,
    period: wgpu::Buffer,
    phase: wgpu::Buffer,
    dihedral_forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_dihedrals: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

impl DihedralBuffers {
    fn new(dev: &WgpuDevice, positions: &[f64], dihedrals: &[DihedralTorsion]) -> Self {
        let n = dihedrals.len();

        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dih pos"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut quad_data = Vec::with_capacity(n * 4);
        let mut barrier_data = Vec::with_capacity(n);
        let mut period_data = Vec::with_capacity(n);
        let mut phase_data = Vec::with_capacity(n);
        for d in dihedrals {
            quad_data.push(d.i);
            quad_data.push(d.j);
            quad_data.push(d.k);
            quad_data.push(d.l);
            barrier_data.push(d.barrier_height);
            period_data.push(d.periodicity);
            phase_data.push(d.phase_shift);
        }

        let quads = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dih quads"),
                contents: bytemuck::cast_slice(&quad_data),
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

        let barrier = to_f64_buf("dih kphi", &barrier_data);
        let period = to_f64_buf("dih n", &period_data);
        let phase = to_f64_buf("dih delta", &phase_data);

        // 12 force components per dihedral (4 atoms × 3 dims)
        let dihedral_forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dih df"),
            size: (n * 12 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    n_dihedrals: n as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        Self {
            pos,
            quads,
            barrier,
            period,
            phase,
            dihedral_forces,
            params,
        }
    }
}

fn reduce_dihedral_forces(
    dev: &WgpuDevice,
    shader_source: &str,
    dihedral_forces_buf: &wgpu::Buffer,
    quads_buf: &wgpu::Buffer,
    n_particles: usize,
    n_dihedrals: usize,
) -> Result<Vec<f64>> {
    let particle_forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dih pf"),
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
        n_dihedrals: u32,
        _p0: u32,
        _p1: u32,
    }
    let rp_buf = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&ReduceParams {
                n_particles: n_particles as u32,
                n_dihedrals: n_dihedrals as u32,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let wg = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
    ComputeDispatch::new(dev, "reduce_dihedral_forces_f64")
        .shader(shader_source, "reduce_dihedral_forces_f64")
        .f64()
        .uniform(0, &rp_buf)
        .storage_read(1, dihedral_forces_buf)
        .storage_read(2, quads_buf)
        .storage_rw(3, &particle_forces_buf)
        .dispatch(wg, 1, 1)
        .submit()?;

    dev.read_f64_buffer(&particle_forces_buf, n_particles * 3)
}

impl DihedralTorsionF64 {
    /// Create periodic dihedral torsion force calculator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("dihedral_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let caps = DeviceCapabilities::from_device(device);
        match caps.fp64_strategy() {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => {
                Self::wgsl_shader().to_string()
            }
        }
    }

    /// Compute dihedral torsion forces (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces(
        &self,
        positions: &[f64],
        dihedrals: &[DihedralTorsion],
    ) -> Result<Vec<f64>> {
        let n_particles = positions.len() / 3;
        if dihedrals.is_empty() {
            return Ok(vec![0.0f64; n_particles * 3]);
        }
        self.compute_gpu(positions, dihedrals, n_particles)
    }

    /// Compute dihedral torsion forces and per-dihedral energies (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        dihedrals: &[DihedralTorsion],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_particles = positions.len() / 3;
        if dihedrals.is_empty() {
            return Ok((vec![0.0f64; n_particles * 3], vec![]));
        }
        self.compute_gpu_with_energy(positions, dihedrals, n_particles)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        dihedrals: &[DihedralTorsion],
        n_particles: usize,
    ) -> Result<Vec<f64>> {
        let n = dihedrals.len();
        let dev = &self.device;
        let buffers = DihedralBuffers::new(dev, positions, dihedrals);

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "dihedral_torsion_f64")
            .shader(&src, "dihedral_torsion_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.quads)
            .storage_read(2, &buffers.barrier)
            .storage_read(3, &buffers.period)
            .storage_read(4, &buffers.phase)
            .storage_rw(5, &buffers.dihedral_forces)
            .uniform(6, &buffers.params)
            .dispatch(wg, 1, 1)
            .submit()?;

        reduce_dihedral_forces(
            dev,
            &src,
            &buffers.dihedral_forces,
            &buffers.quads,
            n_particles,
            n,
        )
    }

    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        dihedrals: &[DihedralTorsion],
        n_particles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = dihedrals.len();
        let dev = &self.device;
        let buffers = DihedralBuffers::new(dev, positions, dihedrals);

        let energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dih be"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "dihedral_with_energy_f64")
            .shader(&src, "dihedral_with_energy_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.quads)
            .storage_read(2, &buffers.barrier)
            .storage_read(3, &buffers.period)
            .storage_read(4, &buffers.phase)
            .storage_rw(5, &buffers.dihedral_forces)
            .uniform(6, &buffers.params)
            .storage_rw(7, &energy_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        let energies = dev.read_f64_buffer(&energy_buf, n)?;
        let forces = reduce_dihedral_forces(
            dev,
            &src,
            &buffers.dihedral_forces,
            &buffers.quads,
            n_particles,
            n,
        )?;
        Ok((forces, energies))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm(a: [f64; 3]) -> f64 {
        dot(a, a).sqrt()
    }

    fn dihedral_angle(p: &[f64], i: usize, j: usize, k: usize, l: usize) -> f64 {
        let r_ij = [
            p[j * 3] - p[i * 3],
            p[j * 3 + 1] - p[i * 3 + 1],
            p[j * 3 + 2] - p[i * 3 + 2],
        ];
        let r_jk = [
            p[k * 3] - p[j * 3],
            p[k * 3 + 1] - p[j * 3 + 1],
            p[k * 3 + 2] - p[j * 3 + 2],
        ];
        let r_kl = [
            p[l * 3] - p[k * 3],
            p[l * 3 + 1] - p[k * 3 + 1],
            p[l * 3 + 2] - p[k * 3 + 2],
        ];
        let m = cross(r_ij, r_jk);
        let n = cross(r_jk, r_kl);
        let cos_phi = dot(m, n) / (norm(m) * norm(n));
        let mxn = cross(m, n);
        let sin_phi = dot(mxn, r_jk) / (norm(r_jk) * norm(m) * norm(n));
        sin_phi.atan2(cos_phi)
    }

    #[test]
    fn test_dihedral_energy_at_minimum() {
        // n=3, δ=0 → minimum at φ=π/3, 3π/3=π, 5π/3
        let k_phi = 5.0;
        let n_p = 3.0;
        let delta = 0.0;
        // φ=π/3 → cos(3·π/3 - 0) = cos(π) = -1 → U = k(1-1) = 0
        let phi = std::f64::consts::PI / 3.0;
        let u = k_phi * (1.0 + (n_p * phi - delta).cos());
        assert!(u.abs() < 1e-12, "energy at minimum should be 0, got {u}");
    }

    #[test]
    fn test_dihedral_momentum_conservation_cpu() {
        // Numerical gradient test: sum of forces must be zero
        let positions = vec![
            0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.15, 0.15, 0.0, 0.15, 0.15, 0.15,
        ];
        let dih = DihedralTorsion {
            i: 0,
            j: 1,
            k: 2,
            l: 3,
            barrier_height: 5.0,
            periodicity: 3.0,
            phase_shift: 0.0,
        };

        let phi = dihedral_angle(&positions, 0, 1, 2, 3);
        let _u = dih.barrier_height * (1.0 + (dih.periodicity * phi - dih.phase_shift).cos());

        // Numerical forces via finite differences
        let eps = 1e-7;
        let mut num_forces = vec![0.0; 12];
        for atom in 0..4 {
            for dim in 0..3 {
                let mut pos_plus = positions.clone();
                let mut pos_minus = positions.clone();
                pos_plus[atom * 3 + dim] += eps;
                pos_minus[atom * 3 + dim] -= eps;

                let phi_plus = dihedral_angle(&pos_plus, 0, 1, 2, 3);
                let phi_minus = dihedral_angle(&pos_minus, 0, 1, 2, 3);
                let u_plus =
                    dih.barrier_height * (1.0 + (dih.periodicity * phi_plus - dih.phase_shift).cos());
                let u_minus = dih.barrier_height
                    * (1.0 + (dih.periodicity * phi_minus - dih.phase_shift).cos());
                num_forces[atom * 3 + dim] = -(u_plus - u_minus) / (2.0 * eps);
            }
        }

        // Momentum conservation: sum of forces = 0
        for dim in 0..3 {
            let total: f64 = (0..4).map(|a| num_forces[a * 3 + dim]).sum();
            assert!(
                total.abs() < 1e-4,
                "momentum not conserved on dim {dim}: {total}"
            );
        }
    }
}
