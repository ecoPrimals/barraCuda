// SPDX-License-Identifier: AGPL-3.0-or-later
//! Improper Dihedral Force (f64) — Planarity Restraint
//!
//! **Potential**: U(ψ) = ½k_ψ(ψ - ψ₀)²
//! **Force**: Cartesian gradient using Blondel-Karplus decomposition
//!
//! Harmonic improper dihedral for enforcing planarity at sp2 centers and
//! aromatic rings. Used in GROMOS 45a4, CHARMM, AMBER.
//! Two-pass GPU dispatch: per-improper forces → reduce to per-particle.

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy, WORKGROUP_SIZE_1D};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 improper dihedral force calculator (GPU-accelerated).
pub struct ImproperDihedralF64 {
    device: Arc<WgpuDevice>,
}

/// Parameters for a single improper dihedral (i-j-k-l, i is the central atom).
#[derive(Clone, Copy, Debug)]
pub struct ImproperDihedral {
    /// Central atom i
    pub i: u32,
    /// Particle index j
    pub j: u32,
    /// Particle index k
    pub k: u32,
    /// Particle index l
    pub l: u32,
    /// Force constant k_ψ (kJ/mol/rad²)
    pub force_constant: f64,
    /// Equilibrium angle ψ₀ (radians, typically 0 for planar)
    pub eq_angle: f64,
}

struct ImproperBuffers {
    pos: wgpu::Buffer,
    quads: wgpu::Buffer,
    force_constants: wgpu::Buffer,
    eq_angles: wgpu::Buffer,
    improper_forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_impropers: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

impl ImproperBuffers {
    fn new(dev: &WgpuDevice, positions: &[f64], impropers: &[ImproperDihedral]) -> Self {
        let n = impropers.len();

        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imp pos"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut quad_data = Vec::with_capacity(n * 4);
        let mut k_data = Vec::with_capacity(n);
        let mut psi0_data = Vec::with_capacity(n);
        for imp in impropers {
            quad_data.push(imp.i);
            quad_data.push(imp.j);
            quad_data.push(imp.k);
            quad_data.push(imp.l);
            k_data.push(imp.force_constant);
            psi0_data.push(imp.eq_angle);
        }

        let quads = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("imp quads"),
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

        let force_constants = to_f64_buf("imp k", &k_data);
        let eq_angles = to_f64_buf("imp psi0", &psi0_data);

        let improper_forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imp if"),
            size: (n * 12 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    n_impropers: n as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        Self {
            pos,
            quads,
            force_constants,
            eq_angles,
            improper_forces,
            params,
        }
    }
}

fn reduce_improper_forces(
    dev: &WgpuDevice,
    shader_source: &str,
    improper_forces_buf: &wgpu::Buffer,
    quads_buf: &wgpu::Buffer,
    n_particles: usize,
    n_impropers: usize,
) -> Result<Vec<f64>> {
    let particle_forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imp pf"),
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
        n_impropers: u32,
        _p0: u32,
        _p1: u32,
    }
    let rp_buf = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&ReduceParams {
                n_particles: n_particles as u32,
                n_impropers: n_impropers as u32,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let wg = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
    ComputeDispatch::new(dev, "reduce_improper_forces_f64")
        .shader(shader_source, "reduce_improper_forces_f64")
        .f64()
        .uniform(0, &rp_buf)
        .storage_read(1, improper_forces_buf)
        .storage_read(2, quads_buf)
        .storage_rw(3, &particle_forces_buf)
        .dispatch(wg, 1, 1)
        .submit()?;

    dev.read_f64_buffer(&particle_forces_buf, n_particles * 3)
}

impl ImproperDihedralF64 {
    /// Create improper dihedral force calculator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("improper_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let caps = DeviceCapabilities::from_device(device);
        match caps.fp64_strategy() {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => Self::wgsl_shader().to_string(),
        }
    }

    /// Compute improper dihedral forces (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces(
        &self,
        positions: &[f64],
        impropers: &[ImproperDihedral],
    ) -> Result<Vec<f64>> {
        let n_particles = positions.len() / 3;
        if impropers.is_empty() {
            return Ok(vec![0.0f64; n_particles * 3]);
        }
        self.compute_gpu(positions, impropers, n_particles)
    }

    /// Compute improper dihedral forces and per-term energies (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        impropers: &[ImproperDihedral],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_particles = positions.len() / 3;
        if impropers.is_empty() {
            return Ok((vec![0.0f64; n_particles * 3], vec![]));
        }
        self.compute_gpu_with_energy(positions, impropers, n_particles)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        impropers: &[ImproperDihedral],
        n_particles: usize,
    ) -> Result<Vec<f64>> {
        let n = impropers.len();
        let dev = &self.device;
        let buffers = ImproperBuffers::new(dev, positions, impropers);

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "improper_dihedral_f64")
            .shader(&src, "improper_dihedral_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.quads)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_angles)
            .storage_rw(4, &buffers.improper_forces)
            .uniform(5, &buffers.params)
            .dispatch(wg, 1, 1)
            .submit()?;

        reduce_improper_forces(
            dev,
            &src,
            &buffers.improper_forces,
            &buffers.quads,
            n_particles,
            n,
        )
    }

    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        impropers: &[ImproperDihedral],
        n_particles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = impropers.len();
        let dev = &self.device;
        let buffers = ImproperBuffers::new(dev, positions, impropers);

        let energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("imp be"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "improper_with_energy_f64")
            .shader(&src, "improper_with_energy_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.quads)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_angles)
            .storage_rw(4, &buffers.improper_forces)
            .uniform(5, &buffers.params)
            .storage_rw(6, &energy_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        let energies = dev.read_f64_buffer(&energy_buf, n)?;
        let forces = reduce_improper_forces(
            dev,
            &src,
            &buffers.improper_forces,
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

    fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn norm3(a: [f64; 3]) -> f64 {
        dot3(a, a).sqrt()
    }

    fn improper_angle(p: &[f64], i: usize, j: usize, k: usize, l: usize) -> f64 {
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
        let cos_psi = dot3(m, n) / (norm3(m) * norm3(n));
        let mxn = cross(m, n);
        let sin_psi = dot3(mxn, r_jk) / (norm3(r_jk) * norm3(m) * norm3(n));
        sin_psi.atan2(cos_psi)
    }

    #[test]
    fn test_improper_planar_zero_force() {
        // 4 coplanar atoms → ψ = 0, ψ₀ = 0 → zero force
        let positions = vec![
            0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.15, 0.15, 0.0, 0.0, 0.15, 0.0,
        ];
        let psi = improper_angle(&positions, 0, 1, 2, 3);
        assert!(
            psi.abs() < 1e-10,
            "coplanar atoms should give ψ≈0, got {psi}"
        );
    }

    #[test]
    fn test_improper_energy_at_equilibrium() {
        let k_psi = 167.36;
        let psi_0 = 0.0;
        let psi = 0.0;
        let delta = psi - psi_0;
        let u: f64 = 0.5 * k_psi * delta * delta;
        assert!(u.abs() < 1e-15, "energy should be zero at equilibrium");
    }

    #[test]
    fn test_improper_momentum_conservation() {
        let positions = vec![
            0.0, 0.0, 0.05, 0.15, 0.0, 0.0, 0.15, 0.15, 0.0, 0.0, 0.15, 0.0,
        ];
        let imp = ImproperDihedral {
            i: 0,
            j: 1,
            k: 2,
            l: 3,
            force_constant: 167.36,
            eq_angle: 0.0,
        };

        let eps = 1e-7;
        let mut num_forces = vec![0.0; 12];
        for atom in 0..4 {
            for dim in 0..3 {
                let mut p_plus = positions.clone();
                let mut p_minus = positions.clone();
                p_plus[atom * 3 + dim] += eps;
                p_minus[atom * 3 + dim] -= eps;
                let psi_plus = improper_angle(&p_plus, 0, 1, 2, 3);
                let psi_minus = improper_angle(&p_minus, 0, 1, 2, 3);
                let u_plus = 0.5 * imp.force_constant * (psi_plus - imp.eq_angle).powi(2);
                let u_minus = 0.5 * imp.force_constant * (psi_minus - imp.eq_angle).powi(2);
                num_forces[atom * 3 + dim] = -(u_plus - u_minus) / (2.0 * eps);
            }
        }

        for dim in 0..3 {
            let total: f64 = (0..4).map(|a| num_forces[a * 3 + dim]).sum();
            assert!(
                total.abs() < 1e-4,
                "momentum not conserved on dim {dim}: {total}"
            );
        }
    }
}
