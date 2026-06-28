// SPDX-License-Identifier: AGPL-3.0-or-later
//! Harmonic Angle Force (f64) — Three-Body Bonded Interaction
//!
//! **Potential**: U(θ) = ½`k_θ`(θ - θ₀)²
//! **Force**: Cartesian gradient of U over atoms i, j (vertex), k
//!
//! Three-body term used by all major biomolecular force fields.
//! Two-pass GPU dispatch: per-angle forces → reduce to per-particle.

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy, WORKGROUP_SIZE_1D};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// f64 harmonic angle force calculator (GPU-accelerated).
pub struct HarmonicAngleF64 {
    device: Arc<WgpuDevice>,
}

/// Parameters for a single angle term (i-j-k, angle at j).
#[derive(Clone, Copy, Debug)]
pub struct HarmonicAngle {
    /// Particle index i
    pub i: u32,
    /// Vertex atom j (angle measured here)
    pub j: u32,
    /// Particle index k
    pub k: u32,
    /// Angular force constant `k_θ` (kJ/mol/rad²)
    pub force_constant: f64,
    /// Equilibrium angle θ₀ (radians)
    pub eq_angle: f64,
}

struct AngleBuffers {
    pos: wgpu::Buffer,
    triples: wgpu::Buffer,
    force_constants: wgpu::Buffer,
    eq_angles: wgpu::Buffer,
    angle_forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_angles: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

impl AngleBuffers {
    fn new(dev: &WgpuDevice, positions: &[f64], angles: &[HarmonicAngle]) -> Self {
        let n_angles = angles.len();

        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("angle pos"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut triple_data = Vec::with_capacity(n_angles * 3);
        let mut k_data = Vec::with_capacity(n_angles);
        let mut theta0_data = Vec::with_capacity(n_angles);
        for a in angles {
            triple_data.push(a.i);
            triple_data.push(a.j);
            triple_data.push(a.k);
            k_data.push(a.force_constant);
            theta0_data.push(a.eq_angle);
        }

        let triples = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("angle triples"),
                contents: bytemuck::cast_slice(&triple_data),
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

        let force_constants = to_f64_buf("angle k", &k_data);
        let eq_angles = to_f64_buf("angle theta0", &theta0_data);

        // 9 force components per angle (3 atoms × 3 dims)
        let angle_forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("angle af"),
            size: (n_angles * 9 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    n_angles: n_angles as u32,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        Self {
            pos,
            triples,
            force_constants,
            eq_angles,
            angle_forces,
            params,
        }
    }
}

fn reduce_angle_forces(
    dev: &WgpuDevice,
    shader_source: &str,
    angle_forces_buf: &wgpu::Buffer,
    triples_buf: &wgpu::Buffer,
    n_particles: usize,
    n_angles: usize,
) -> Result<Vec<f64>> {
    let particle_forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("angle pf"),
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
        n_angles: u32,
        _p0: u32,
        _p1: u32,
    }
    let rp_buf = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&ReduceParams {
                n_particles: n_particles as u32,
                n_angles: n_angles as u32,
                _p0: 0,
                _p1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let wg = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
    ComputeDispatch::new(dev, "reduce_angle_forces_f64")
        .shader(shader_source, "reduce_angle_forces_f64")
        .f64()
        .uniform(0, &rp_buf)
        .storage_read(1, angle_forces_buf)
        .storage_read(2, triples_buf)
        .storage_rw(3, &particle_forces_buf)
        .dispatch(wg, 1, 1)
        .submit()?;

    dev.read_f64_buffer(&particle_forces_buf, n_particles * 3)
}

impl HarmonicAngleF64 {
    /// Create harmonic angle force calculator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("harmonic_angle_f64.wgsl")
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

    /// Compute harmonic angle forces for all angles (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces(&self, positions: &[f64], angles: &[HarmonicAngle]) -> Result<Vec<f64>> {
        let n_particles = positions.len() / 3;
        if angles.is_empty() {
            return Ok(vec![0.0f64; n_particles * 3]);
        }
        self.compute_gpu(positions, angles, n_particles)
    }

    /// Compute harmonic angle forces and per-angle energies (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] on GPU dispatch or readback failure.
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        angles: &[HarmonicAngle],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_particles = positions.len() / 3;
        if angles.is_empty() {
            return Ok((vec![0.0f64; n_particles * 3], vec![]));
        }
        self.compute_gpu_with_energy(positions, angles, n_particles)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        angles: &[HarmonicAngle],
        n_particles: usize,
    ) -> Result<Vec<f64>> {
        let n_angles = angles.len();
        let dev = &self.device;
        let buffers = AngleBuffers::new(dev, positions, angles);

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_angles as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "harmonic_angle_f64")
            .shader(&src, "harmonic_angle_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.triples)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_angles)
            .storage_rw(4, &buffers.angle_forces)
            .uniform(5, &buffers.params)
            .dispatch(wg, 1, 1)
            .submit()?;

        reduce_angle_forces(
            dev,
            &src,
            &buffers.angle_forces,
            &buffers.triples,
            n_particles,
            n_angles,
        )
    }

    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        angles: &[HarmonicAngle],
        n_particles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_angles = angles.len();
        let dev = &self.device;
        let buffers = AngleBuffers::new(dev, positions, angles);

        let energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("angle be"),
            size: (n_angles * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_angles as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "harmonic_angle_with_energy_f64")
            .shader(&src, "harmonic_angle_with_energy_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.triples)
            .storage_read(2, &buffers.force_constants)
            .storage_read(3, &buffers.eq_angles)
            .storage_rw(4, &buffers.angle_forces)
            .uniform(5, &buffers.params)
            .storage_rw(6, &energy_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        let energies = dev.read_f64_buffer(&energy_buf, n_angles)?;
        let forces = reduce_angle_forces(
            dev,
            &src,
            &buffers.angle_forces,
            &buffers.triples,
            n_particles,
            n_angles,
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

    fn compute_cpu(positions: &[f64], angles: &[HarmonicAngle]) -> Vec<f64> {
        let n_particles = positions.len() / 3;
        let mut forces = vec![0.0f64; n_particles * 3];
        for a in angles {
            let (i, j, k) = (a.i as usize, a.j as usize, a.k as usize);
            let rji = [
                positions[i * 3] - positions[j * 3],
                positions[i * 3 + 1] - positions[j * 3 + 1],
                positions[i * 3 + 2] - positions[j * 3 + 2],
            ];
            let rjk = [
                positions[k * 3] - positions[j * 3],
                positions[k * 3 + 1] - positions[j * 3 + 1],
                positions[k * 3 + 2] - positions[j * 3 + 2],
            ];
            let rji_len = (rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2]).sqrt();
            let rjk_len = (rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2]).sqrt();
            if rji_len < 1e-10 || rjk_len < 1e-10 {
                continue;
            }
            let cos_theta =
                (rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2]) / (rji_len * rjk_len);
            let cos_clamped = cos_theta.clamp(-1.0, 1.0);
            let theta = cos_clamped.acos();
            let sin_theta = (1.0 - cos_clamped * cos_clamped).sqrt().max(1e-12);

            let prefactor = -a.force_constant * (theta - a.eq_angle) / sin_theta;

            let rji_inv = 1.0 / rji_len;
            let rjk_inv = 1.0 / rjk_len;
            let rji_inv2 = rji_inv * rji_inv;
            let rjk_inv2 = rjk_inv * rjk_inv;

            for d in 0..3 {
                let fi = prefactor * (rjk[d] * rji_inv * rjk_inv - cos_clamped * rji[d] * rji_inv2);
                let fk = prefactor * (rji[d] * rji_inv * rjk_inv - cos_clamped * rjk[d] * rjk_inv2);
                forces[i * 3 + d] += fi;
                forces[k * 3 + d] += fk;
                forces[j * 3 + d] += -(fi + fk);
            }
        }
        forces
    }

    #[test]
    fn test_harmonic_angle_equilibrium_cpu() {
        let theta0: f64 = std::f64::consts::FRAC_PI_3 * 2.0; // 120°
        let r = 0.15_f64;
        let positions = vec![
            r * (theta0 / 2.0).cos(),
            r * (theta0 / 2.0).sin(),
            0.0,
            0.0,
            0.0,
            0.0,
            r * (theta0 / 2.0).cos(),
            -r * (theta0 / 2.0).sin(),
            0.0,
        ];
        let angles = vec![HarmonicAngle {
            i: 0,
            j: 1,
            k: 2,
            force_constant: 500.0,
            eq_angle: theta0,
        }];
        let forces = compute_cpu(&positions, &angles);
        for f in &forces {
            assert!(
                f.abs() < 1e-8,
                "force at equilibrium should be ~zero, got {f}"
            );
        }
    }

    #[test]
    fn test_harmonic_angle_momentum_conservation_cpu() {
        let positions = vec![0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.05];
        let angles = vec![HarmonicAngle {
            i: 0,
            j: 1,
            k: 2,
            force_constant: 500.0,
            eq_angle: std::f64::consts::FRAC_PI_2,
        }];
        let forces = compute_cpu(&positions, &angles);
        for d in 0..3 {
            let total = forces[d] + forces[3 + d] + forces[6 + d];
            assert!(
                total.abs() < 1e-10,
                "momentum not conserved on dim {d}: {total}"
            );
        }
    }
}
