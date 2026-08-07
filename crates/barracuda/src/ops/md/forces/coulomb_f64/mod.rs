// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coulomb Force Calculation (f64)
//!
//! **Physics**: Electrostatic interactions between charged particles
//! **Formula**: F = k * `q_i` * `q_j` / r² * r̂
//! **Use Case**: Ions, proteins, charged molecules, nuclei
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader (f64)
//! - ✅ Zero unsafe code
//! - ✅ Capability-based dispatch
//! - ✅ Agnostic (no hardcoded constants)
//!
//! **Precision**: f64 is critical for:
//! - Large systems where small forces accumulate
//! - Nuclear physics (fine structure constant precision)
//! - Long timescale simulations

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

#[cfg(test)]
mod cpu_reference;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CoulombParams {
    n_particles: u32,
    _pad0: u32,
    coulomb_constant: f64,
    cutoff_radius: f64,
    cutoff_radius_sq: f64,
    softening: f64,
}

/// Shared GPU buffers for Coulomb calculations.
struct CoulombBuffers {
    pos: wgpu::Buffer,
    charges: wgpu::Buffer,
    forces: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl CoulombBuffers {
    fn new(
        dev: &WgpuDevice,
        positions: &[f64],
        charges: &[f64],
        k: f64,
        cutoff: f64,
        eps: f64,
    ) -> Self {
        let n = charges.len();
        let pos = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Coulomb f64 Positions"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let charges_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Coulomb f64 Charges"),
                contents: bytemuck::cast_slice(charges),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let forces = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Coulomb f64 Forces"),
            size: (n * 3 * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Coulomb f64 Params"),
                contents: bytemuck::cast_slice(&[CoulombParams {
                    n_particles: n as u32,
                    _pad0: 0,
                    coulomb_constant: k,
                    cutoff_radius: cutoff,
                    cutoff_radius_sq: cutoff * cutoff,
                    softening: eps,
                }]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        Self {
            pos,
            charges: charges_buf,
            forces,
            params,
        }
    }
}

/// f64 Coulomb force calculation operation
///
/// Computes electrostatic forces between all particle pairs.
/// Uses softened potential to avoid singularities.
pub struct CoulombForceF64 {
    device: Arc<WgpuDevice>,
}

impl CoulombForceF64 {
    /// Create new Coulomb f64 force calculation
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("coulomb_f64.wgsl")
    }

    /// Execute Coulomb force calculation
    /// # Arguments
    /// * `positions` - Particle positions [N*3] (x,y,z interleaved)
    /// * `charges` - Particle charges [N]
    /// * `coulomb_constant` - Coulomb constant k (default: 1.0)
    /// * `cutoff_radius` - Cutoff distance (default: infinity)
    /// * `softening` - Softening parameter (default: 1e-10)
    /// # Returns
    /// Force vectors [N*3] containing force for each particle
    /// # Errors
    /// Returns [`Err`] if `positions.len() != 3 * charges.len()`, or if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn compute_forces(
        &self,
        positions: &[f64],
        charges: &[f64],
        coulomb_constant: Option<f64>,
        cutoff_radius: Option<f64>,
        softening: Option<f64>,
    ) -> Result<Vec<f64>> {
        let n = charges.len();
        if positions.len() != n * 3 {
            return Err(BarracudaError::invalid_input(format!(
                "Position length {} != 3 * charges length {}",
                positions.len(),
                n * 3
            )));
        }

        self.compute_gpu(
            positions,
            charges,
            coulomb_constant.unwrap_or(1.0),
            cutoff_radius.unwrap_or(f64::INFINITY),
            softening.unwrap_or(1e-10),
            "coulomb_f64",
        )
    }

    /// Compute forces with potential energy output
    /// # Errors
    /// Returns [`Err`] if `positions.len() != 3 * charges.len()`, or if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn compute_forces_and_energy(
        &self,
        positions: &[f64],
        charges: &[f64],
        coulomb_constant: Option<f64>,
        cutoff_radius: Option<f64>,
        softening: Option<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = charges.len();
        if positions.len() != n * 3 {
            return Err(BarracudaError::invalid_input(format!(
                "Position length {} != 3 * charges length {}",
                positions.len(),
                n * 3
            )));
        }

        let k = coulomb_constant.unwrap_or(1.0);
        let cutoff = cutoff_radius.unwrap_or(f64::INFINITY);
        let eps = softening.unwrap_or(1e-10);

        self.compute_gpu_with_energy(positions, charges, k, cutoff, eps)
    }

    fn compute_gpu(
        &self,
        positions: &[f64],
        charges: &[f64],
        k: f64,
        cutoff: f64,
        eps: f64,
        entry_point: &str,
    ) -> Result<Vec<f64>> {
        let n = charges.len();
        let dev = &self.device;
        let bufs = CoulombBuffers::new(dev, positions, charges, k, cutoff, eps);

        ComputeDispatch::new(dev, "Coulomb f64")
            .shader(Self::wgsl_shader(), entry_point)
            .f64()
            .storage_read(0, &bufs.pos)
            .storage_read(1, &bufs.charges)
            .storage_rw(2, &bufs.forces)
            .uniform(3, &bufs.params)
            .dispatch_1d(n as u32)
            .submit()?;

        dev.read_buffer_f64(&bufs.forces, n * 3)
    }

    fn compute_gpu_with_energy(
        &self,
        positions: &[f64],
        charges: &[f64],
        k: f64,
        cutoff: f64,
        eps: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = charges.len();
        let dev = &self.device;
        let bufs = CoulombBuffers::new(dev, positions, charges, k, cutoff, eps);

        let energy_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Coulomb f64 Energy"),
            size: std::mem::size_of_val(charges) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(dev, "Coulomb f64 Energy")
            .shader(Self::wgsl_shader(), "coulomb_with_energy_f64")
            .f64()
            .storage_read(0, &bufs.pos)
            .storage_read(1, &bufs.charges)
            .storage_rw(2, &bufs.forces)
            .uniform(3, &bufs.params)
            .storage_rw(4, &energy_buf)
            .dispatch_1d(n as u32)
            .submit()?;

        let forces = dev.read_buffer_f64(&bufs.forces, n * 3)?;
        let energies = dev.read_buffer_f64(&energy_buf, n)?;
        Ok((forces, energies))
    }
}
