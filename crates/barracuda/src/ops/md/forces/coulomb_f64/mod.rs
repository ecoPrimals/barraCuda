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
use crate::device::capabilities::WORKGROUP_SIZE_1D;
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

/// Map a GPU buffer back to CPU as `Vec<f64>`.
fn read_f64_via_staging(
    dev: &WgpuDevice,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::Buffer,
    count: usize,
    label: &str,
) -> (wgpu::Buffer, usize) {
    let size = (count * std::mem::size_of::<f64>()) as u64;
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    (staging, count)
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
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Position length {} != 3 * charges length {}",
                    positions.len(),
                    n * 3
                ),
            });
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
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Position length {} != 3 * charges length {}",
                    positions.len(),
                    n * 3
                ),
            });
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

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Coulomb f64 BGL"),
                entries: &(0..4u32)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: match i {
                                3 => wgpu::BufferBindingType::Uniform,
                                2 => wgpu::BufferBindingType::Storage { read_only: false },
                                _ => wgpu::BufferBindingType::Storage { read_only: true },
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });

        let bind_bufs: [&wgpu::Buffer; 4] = [&bufs.pos, &bufs.charges, &bufs.forces, &bufs.params];
        let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coulomb f64 Bind Group"),
            layout: &bgl,
            entries: &bind_bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("Coulomb f64"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coulomb f64 Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Coulomb f64 Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }

        let (staging, count) =
            read_f64_via_staging(dev, &mut encoder, &bufs.forces, n * 3, "Coulomb Staging");
        dev.submit_commands(Some(encoder.finish()));
        dev.map_staging_buffer::<f64>(&staging, count)
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

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Coulomb f64 Energy BGL"),
                entries: &(0..5u32)
                    .map(|i| wgpu::BindGroupLayoutEntry {
                        binding: i,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: match i {
                                3 => wgpu::BufferBindingType::Uniform,
                                0 | 1 => wgpu::BufferBindingType::Storage { read_only: true },
                                _ => wgpu::BufferBindingType::Storage { read_only: false },
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });

        let bind_bufs: [&wgpu::Buffer; 5] = [
            &bufs.pos,
            &bufs.charges,
            &bufs.forces,
            &bufs.params,
            &energy_buf,
        ];
        let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Coulomb f64 Energy Bind Group"),
            layout: &bgl,
            entries: &bind_bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("Coulomb f64 Energy"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Coulomb f64 Energy Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("coulomb_with_energy_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Coulomb f64 Energy Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }

        let (forces_stg, forces_count) = read_f64_via_staging(
            dev,
            &mut encoder,
            &bufs.forces,
            n * 3,
            "Coulomb Forces Staging",
        );
        let (energy_stg, energy_count) =
            read_f64_via_staging(dev, &mut encoder, &energy_buf, n, "Coulomb Energy Staging");
        dev.submit_commands(Some(encoder.finish()));

        let forces = dev.map_staging_buffer::<f64>(&forces_stg, forces_count)?;
        let energies = dev.map_staging_buffer::<f64>(&energy_stg, energy_count)?;
        Ok((forces, energies))
    }
}
