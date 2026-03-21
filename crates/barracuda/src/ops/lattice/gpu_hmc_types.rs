// SPDX-License-Identifier: AGPL-3.0-or-later
//! Types, configuration, and buffer management for GPU HMC trajectories.

use super::constants;
use super::gpu_cg_solver::GpuCgBuffers;
use crate::device::WgpuDevice;
use crate::device::capabilities::DeviceCapabilities;
use crate::error::Result;
use wgpu;

/// Configuration for a GPU HMC trajectory.
#[derive(Clone, Debug)]
pub struct GpuHmcConfig {
    /// Temporal lattice extent.
    pub nt: u32,
    /// Spatial lattice extent (x).
    pub nx: u32,
    /// Spatial lattice extent (y).
    pub ny: u32,
    /// Spatial lattice extent (z).
    pub nz: u32,
    /// Gauge coupling β = 6/g².
    pub beta: f64,
    /// Staggered fermion mass.
    pub mass: f64,
    /// Number of molecular dynamics steps per trajectory.
    pub n_md_steps: usize,
    /// MD step size (leapfrog dt).
    pub dt: f64,
    /// Conjugate gradient tolerance for pseudofermion inversion.
    pub cg_tol: f64,
    /// Maximum CG iterations.
    pub cg_max_iter: usize,
    /// Number of fermion flavors / 4 (e.g. 2 for 8 flavors).
    pub n_flavors_over_4: usize,
}

impl Default for GpuHmcConfig {
    fn default() -> Self {
        Self {
            nt: 4,
            nx: 4,
            ny: 4,
            nz: 4,
            beta: 5.5,
            mass: 0.1,
            n_md_steps: 20,
            dt: 0.02,
            cg_tol: constants::CG_TOL_DEFAULT,
            cg_max_iter: constants::CG_MAX_ITER_DEFAULT,
            n_flavors_over_4: 2,
        }
    }
}

/// Result of a GPU HMC trajectory.
#[derive(Clone, Debug)]
pub struct GpuHmcResult {
    /// Whether the Metropolis step accepted the new configuration.
    pub accepted: bool,
    /// Change in Hamiltonian `H_new` − `H_old`.
    pub delta_h: f64,
    /// Wilson gauge action `S_G` = β × (sum of plaquettes).
    pub gauge_action: f64,
    /// Fermion action `S_F` = Σ φ†(D†D)⁻¹φ.
    pub fermion_action: f64,
    /// Kinetic energy of link momenta.
    pub kinetic_energy: f64,
    /// Total CG iterations across the trajectory.
    pub total_cg_iterations: usize,
}

/// GPU-resident buffer set for the full HMC trajectory.
pub struct GpuHmcBuffers {
    /// SU(3) gauge links (4 directions × volume).
    pub links: wgpu::Buffer,
    /// Backup of links for Metropolis reject rollback.
    pub links_backup: wgpu::Buffer,
    /// Canonical momenta conjugate to links.
    pub momenta: wgpu::Buffer,
    /// Gauge force from Wilson plaquettes.
    pub gauge_force: wgpu::Buffer,
    /// Fermion force from pseudofermion determinant.
    pub fermion_force: wgpu::Buffer,
    /// Gauge + fermion force (accumulated).
    pub total_force: wgpu::Buffer,
    /// Wilson action per site (for reduction).
    pub action_per_site: wgpu::Buffer,
    /// Kinetic energy per link (for reduction).
    pub energy_per_link: wgpu::Buffer,
    /// RNG seeds for link momenta generation.
    pub rng_links: wgpu::Buffer,
    /// RNG seeds for pseudofermion heatbath.
    pub rng_sites: wgpu::Buffer,
    /// Neighbor indices (staggered Dirac).
    pub nbr: wgpu::Buffer,
    /// Staggered phases.
    pub phases: wgpu::Buffer,
    /// Pseudofermion fields φ = (D†)⁻¹η.
    pub phi_fields: Vec<wgpu::Buffer>,
    /// Gaussian noise η for heatbath.
    pub eta: wgpu::Buffer,
    /// Temporary buffer for Dirac application.
    pub dirac_tmp: wgpu::Buffer,
    /// Conjugate gradient solver buffers.
    pub cg: GpuCgBuffers,
}

impl GpuHmcBuffers {
    /// Allocate all GPU buffers for the given HMC config.
    /// # Errors
    /// Returns [`Err`] if the estimated allocation exceeds driver limits, buffer allocation fails, or the device is lost.
    pub fn new(device: &WgpuDevice, config: &GpuHmcConfig) -> Result<Self> {
        let volume = (config.nt * config.nx * config.ny * config.nz) as usize;
        let n_links = volume * 4;
        let link_bytes = (n_links * 18 * std::mem::size_of::<f64>()) as u64;
        let field_bytes = (volume * 6 * std::mem::size_of::<f64>()) as u64;

        // NVK buffer guard: estimate total allocation and check driver limits
        let n_link_bufs = 6u64; // links, backup, momenta, gauge/fermion/total force
        let n_field_bufs = 2 + config.n_flavors_over_4 as u64 + 5; // phi + eta + dirac_tmp + CG bufs
        let scalar_bufs = (volume as u64 + n_links as u64 + volume as u64 * 2)
            * std::mem::size_of::<u32>() as u64;
        let total_estimate = n_link_bufs * link_bytes + n_field_bufs * field_bytes + scalar_bufs;

        let caps = DeviceCapabilities::from_device(device);
        if let Some(limit) = caps.max_safe_allocation_bytes() {
            if total_estimate > limit {
                return Err(crate::error::BarracudaError::DeviceLimitExceeded {
                    message: format!(
                        "Estimated allocation {:.1} MB exceeds safe limit {:.1} MB",
                        total_estimate as f64 / 1e6,
                        limit as f64 / 1e6,
                    ),
                    requested_bytes: total_estimate,
                    safe_limit_bytes: limit,
                });
            }
        }

        let make_link_buf = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: link_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let make_field_buf = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: field_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let phi_fields = (0..config.n_flavors_over_4)
            .map(|i| make_field_buf(&format!("hmc:phi_{i}")))
            .collect();

        Ok(Self {
            links: make_link_buf("hmc:links"),
            links_backup: make_link_buf("hmc:links_backup"),
            momenta: make_link_buf("hmc:momenta"),
            gauge_force: make_link_buf("hmc:gauge_force"),
            fermion_force: make_link_buf("hmc:fermion_force"),
            total_force: make_link_buf("hmc:total_force"),
            action_per_site: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:action_per_site"),
                size: (volume * std::mem::size_of::<f64>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            energy_per_link: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:energy_per_link"),
                size: (n_links * std::mem::size_of::<f64>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            rng_links: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:rng_links"),
                size: (n_links * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            rng_sites: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:rng_sites"),
                size: (volume * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            nbr: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:nbr"),
                size: (volume * 8 * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            phases: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hmc:phases"),
                size: (volume * 4 * std::mem::size_of::<f64>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            phi_fields,
            eta: make_field_buf("hmc:eta"),
            dirac_tmp: make_field_buf("hmc:dirac_tmp"),
            cg: GpuCgBuffers::new(device, volume),
        })
    }
}

/// WGSL uniform params for complex dot product (`n_pairs` layout).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[expect(
    missing_docs,
    reason = "GPU pipeline types follow WGSL struct layout, not public API"
)]
pub struct DotParamsLocal {
    pub n_pairs: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// WGSL uniform params for axpy (`y = alpha*x + y`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[expect(
    missing_docs,
    reason = "GPU pipeline types follow WGSL struct layout, not public API"
)]
pub struct AxpyParamsLocal {
    pub n: u32,
    pub pad0: u32,
    pub alpha: f64,
}

/// Seeded host-side PRNG for Metropolis accept/reject.
///
/// Uses the lattice LCG (Knuth MMIX) with a mutable seed that the caller
/// advances across trajectories for reproducible accept/reject decisions.
pub struct HostRng {
    state: u64,
}

impl HostRng {
    /// Create a new RNG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Draw a uniform f64 in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        constants::lcg_uniform_f64(&mut self.state)
    }
}

/// Create a storage buffer bind group layout entry.
#[must_use]
pub fn storage_bgl(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a uniform buffer bind group layout entry.
#[must_use]
pub fn uniform_bgl(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
