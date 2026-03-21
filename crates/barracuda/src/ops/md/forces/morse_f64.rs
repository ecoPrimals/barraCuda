// SPDX-License-Identifier: AGPL-3.0-or-later
//! Morse Force Calculation (f64)
//!
//! **Physics**: Anharmonic bonded interactions (chemical bonds)
//! **Potential**: U(r) = D·[1 - exp(-a(r-r₀))]²
//! **Use Case**: Molecular mechanics, reactive MD, bond stretching/breaking
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader (f64)
//! - ✅ Zero unsafe code
//! - ✅ Capability-based dispatch
//! - ✅ Agnostic (no hardcoded constants)

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy, WORKGROUP_SIZE_1D};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

const WGSL_DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
const WGSL_DF64_TRANSCENDENTALS: &str =
    include_str!("../../../shaders/math/df64_transcendentals.wgsl");
const MORSE_SHADER_DF64: &str = include_str!("morse_df64.wgsl");

/// f64 Morse force calculation for bonded interactions
///
/// Computes forces and energies for chemical bonds using Morse potential.
/// GPU shader dispatch via `morse_f64.wgsl` (2-pass: force + energy).
pub struct MorseForceF64 {
    device: Arc<WgpuDevice>,
}

/// Parameters for a single Morse bond
#[derive(Clone, Copy, Debug)]
pub struct MorseBond {
    /// Particle index i
    pub i: u32,
    /// Particle index j
    pub j: u32,
    /// Dissociation energy D (eV or kJ/mol)
    pub dissociation_energy: f64,
    /// Width parameter a (1/Å or 1/nm)
    pub width_param: f64,
    /// Equilibrium bond distance r₀ (Å or nm)
    pub equilibrium_dist: f64,
}

/// Shared GPU buffers for Morse bond calculations.
///
/// Extracted to eliminate duplication between `compute_gpu` and `compute_gpu_with_energy`.
struct MorseBuffers {
    pos: wgpu::Buffer,
    pairs: wgpu::Buffer,
    de: wgpu::Buffer,
    wp: wgpu::Buffer,
    eq: wgpu::Buffer,
    bond_forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MorseParams {
    n_bonds: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

impl MorseBuffers {
    fn new(dev: &WgpuDevice, positions: &[f64], bonds: &[MorseBond]) -> Self {
        let n_bonds = bonds.len();

        let pos_bytes: &[u8] = bytemuck::cast_slice(positions);
        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("morse pos"),
                contents: pos_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut pair_data = Vec::with_capacity(n_bonds * 2);
        let mut de_data = Vec::with_capacity(n_bonds);
        let mut wp_data = Vec::with_capacity(n_bonds);
        let mut eq_data = Vec::with_capacity(n_bonds);
        for b in bonds {
            pair_data.push(b.i);
            pair_data.push(b.j);
            de_data.push(b.dissociation_energy);
            wp_data.push(b.width_param);
            eq_data.push(b.equilibrium_dist);
        }

        let pairs = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("morse pairs"),
                contents: bytemuck::cast_slice(&pair_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let to_f64_buf = |label: &str, data: &[f64]| -> wgpu::Buffer {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            dev.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let de = to_f64_buf("morse de", &de_data);
        let wp = to_f64_buf("morse wp", &wp_data);
        let eq = to_f64_buf("morse eq", &eq_data);

        let bond_forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morse bf"),
            size: (n_bonds * 6 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&MorseParams {
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
            de,
            wp,
            eq,
            bond_forces,
            params,
        }
    }
}

/// Reduce per-bond forces to per-particle forces using the `reduce_bond_forces_f64` entry point.
///
/// Shared between `compute_gpu` and `compute_gpu_with_energy`.
fn reduce_bond_forces(
    dev: &WgpuDevice,
    shader_source: &str,
    bond_forces_buf: &wgpu::Buffer,
    pairs_buf: &wgpu::Buffer,
    n_particles: usize,
    n_bonds: usize,
) -> Result<Vec<f64>> {
    let particle_forces_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("morse pf"),
        size: (n_particles * 3 * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc_clear = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
    enc_clear.clear_buffer(&particle_forces_buf, 0, None);
    dev.submit_commands(Some(enc_clear.finish()));

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
    ComputeDispatch::new(dev, "reduce_bond_forces_f64")
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

impl MorseForceF64 {
    /// Create Morse potential force calculator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("morse_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let caps = DeviceCapabilities::from_device(device);
        let strategy = caps.fp64_strategy();
        tracing::info!(?strategy, "Morse F64: using {:?} FP64 strategy", strategy);
        match strategy {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => {
                format!("{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{MORSE_SHADER_DF64}")
            }
        }
    }

    /// Compute Morse forces for all bonds (always GPU dispatch).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn compute_forces(&self, positions: &[f64], bonds: &[MorseBond]) -> Result<Vec<f64>> {
        let n_particles = positions.len() / 3;
        if bonds.is_empty() {
            return Ok(vec![0.0f64; n_particles * 3]);
        }
        self.compute_gpu(positions, bonds, n_particles)
    }

    /// Compute Morse forces and per-bond energies (GPU dispatch).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        bonds: &[MorseBond],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_particles = positions.len() / 3;
        if bonds.is_empty() {
            return Ok((vec![0.0f64; n_particles * 3], vec![]));
        }
        self.compute_gpu_with_energy(positions, bonds, n_particles)
    }

    /// GPU 2-pass with energy: (1) per-bond forces+energy, (2) reduce to per-particle
    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        bonds: &[MorseBond],
        n_particles: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n_bonds = bonds.len();
        let dev = &self.device;
        let buffers = MorseBuffers::new(dev, positions, bonds);

        let bond_energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("morse be"),
            size: (n_bonds * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_bonds as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "morse_with_energy_f64")
            .shader(&src, "morse_with_energy_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.pairs)
            .storage_read(2, &buffers.de)
            .storage_read(3, &buffers.wp)
            .storage_read(4, &buffers.eq)
            .storage_rw(5, &buffers.bond_forces)
            .uniform(6, &buffers.params)
            .storage_rw(7, &bond_energy_buf)
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

    /// GPU 2-pass: (1) per-bond forces, (2) reduce to per-particle
    fn compute_gpu(
        &self,
        positions: &[f64],
        bonds: &[MorseBond],
        n_particles: usize,
    ) -> Result<Vec<f64>> {
        let n_bonds = bonds.len();
        let dev = &self.device;
        let buffers = MorseBuffers::new(dev, positions, bonds);

        let src = Self::wgsl_shader_for_device(dev);
        let wg = (n_bonds as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "morse_bonds_f64")
            .shader(&src, "morse_bonds_f64")
            .f64()
            .storage_read(0, &buffers.pos)
            .storage_read(1, &buffers.pairs)
            .storage_read(2, &buffers.de)
            .storage_read(3, &buffers.wp)
            .storage_read(4, &buffers.eq)
            .storage_rw(5, &buffers.bond_forces)
            .uniform(6, &buffers.params)
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
}

#[cfg(test)]
#[path = "morse_f64_tests.rs"]
mod tests;
