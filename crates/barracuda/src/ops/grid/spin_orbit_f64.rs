// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spin-Orbit Coupling (f64) — GPU-Accelerated HFB Term
//!
//! Computes diagonal spin-orbit corrections to the HFB Hamiltonian.
//!
//! **Evolution**: Feb 16, 2026 — TIER 2.1 from hotSpring handoff
//!
//! This was previously computed on CPU after GPU H-build readback.
//! Moving it to GPU eliminates the last CPU physics in the H-build path.
//!
//! ## Formula
//!
//! ```text
//! H_so[i,i] += w0 · ls_i · ∫ |ψ_i(r)|² · (dρ/dr) · r · dr
//! ```
//!
//! where `ls_i = (j(j+1) - l(l+1) - 3/4) / 2` is the spin-orbit factor.
//!
//! ## Usage
//!
//! ```rust,ignore
//! let so = SpinOrbitGpu::new(device, batch_size, n_states, n_grid)?;
//!
//! // Method 1: Pre-computed gradient
//! let h_so = so.compute(&wf_squared, &drho_dr, &r_grid, &ls_factors, dr, w0)?;
//!
//! // Method 2: Compute gradient internally from density
//! let h_so = so.compute_with_density(&wf_squared, &density, &r_grid, &ls_factors, dr, w0)?;
//! ```

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for spin-orbit shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpinOrbitParams {
    batch_size: u32,
    n_states: u32,
    n_grid: u32,
    _pad: u32,
    dr_lo: u32,
    dr_hi: u32,
    w0_lo: u32,
    w0_hi: u32,
}

impl SpinOrbitParams {
    fn new(batch_size: u32, n_states: u32, n_grid: u32, dr: f64, w0: f64) -> Self {
        let dr_bits = dr.to_bits();
        let w0_bits = w0.to_bits();
        Self {
            batch_size,
            n_states,
            n_grid,
            _pad: 0,
            dr_lo: dr_bits as u32,
            dr_hi: (dr_bits >> 32) as u32,
            w0_lo: w0_bits as u32,
            w0_hi: (w0_bits >> 32) as u32,
        }
    }
}

/// Input arrays and physics parameters for a spin-orbit computation.
pub struct SpinOrbitInputs<'a> {
    /// Squared wave-function amplitudes `[batch_size × n_states × n_grid]`.
    pub wf_squared: &'a [f64],
    /// Radial density gradient (for diagonal mode).
    pub drho_dr: Option<&'a [f64]>,
    /// Electron density (for gradient mode — gradient computed internally).
    pub density: Option<&'a [f64]>,
    /// Radial grid points `[n_grid]`.
    pub r_grid: &'a [f64],
    /// Angular momentum (l·s) factors per state `[n_states]`.
    pub ls_factors: &'a [f64],
    /// Number of independent systems in the batch.
    pub batch_size: usize,
    /// Number of orbital states.
    pub n_states: usize,
    /// Number of radial grid points.
    pub n_grid: usize,
    /// Radial grid spacing.
    pub dr: f64,
    /// Spin-orbit coupling strength W₀.
    pub w0: f64,
    /// Shader entry point (`"spin_orbit_diagonal"` or `"spin_orbit_with_gradient"`).
    pub entry_point: &'a str,
}

/// GPU-accelerated spin-orbit coupling computation
pub struct SpinOrbitGpu {
    device: Arc<WgpuDevice>,
}

impl SpinOrbitGpu {
    /// Create a new spin-orbit GPU operator
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/grid/spin_orbit_f64.wgsl")
    }

    /// Compute spin-orbit diagonal corrections with pre-computed gradient
    ///
    /// # Arguments
    /// * `wf_squared` - Squared wavefunctions [batch × `n_states` × `n_grid`]
    /// * `drho_dr` - Density gradient [batch × `n_grid`]
    /// * `r_grid` - Radial grid points [`n_grid`]
    /// * `ls_factors` - Spin-orbit factors `ls_i` [batch × `n_states`]
    /// * `dr` - Grid spacing
    /// * `w0` - Spin-orbit coupling strength (MeV·fm⁵)
    ///
    /// # Returns
    /// Diagonal corrections `h_so`[i,i] for each (batch, state) pair [batch × `n_states`]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(
        &self,
        wf_squared: &[f64],
        drho_dr: &[f64],
        r_grid: &[f64],
        ls_factors: &[f64],
        dr: f64,
        w0: f64,
    ) -> Result<Vec<f64>> {
        let n_grid = r_grid.len();
        if n_grid == 0 {
            return Err(BarracudaError::invalid_input("r_grid cannot be empty"));
        }

        // Infer batch_size and n_states from ls_factors and drho_dr
        let batch_size = drho_dr.len() / n_grid;
        if drho_dr.len() != batch_size * n_grid {
            return Err(BarracudaError::invalid_input(
                "drho_dr length must be batch_size × n_grid",
            ));
        }

        let n_states = ls_factors.len() / batch_size;
        if ls_factors.len() != batch_size * n_states {
            return Err(BarracudaError::invalid_input(
                "ls_factors length must be batch_size × n_states",
            ));
        }

        if wf_squared.len() != batch_size * n_states * n_grid {
            return Err(BarracudaError::invalid_input(format!(
                "wf_squared length {} must be batch({}) × n_states({}) × n_grid({})",
                wf_squared.len(),
                batch_size,
                n_states,
                n_grid
            )));
        }

        self.compute_internal(&SpinOrbitInputs {
            wf_squared,
            drho_dr: Some(drho_dr),
            density: None,
            r_grid,
            ls_factors,
            batch_size,
            n_states,
            n_grid,
            dr,
            w0,
            entry_point: "spin_orbit_diagonal",
        })
    }

    /// Compute spin-orbit corrections with internal gradient computation
    ///
    /// This version takes density and computes the gradient internally,
    /// which is more efficient when density is already on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute_with_density(
        &self,
        wf_squared: &[f64],
        density: &[f64],
        r_grid: &[f64],
        ls_factors: &[f64],
        dr: f64,
        w0: f64,
    ) -> Result<Vec<f64>> {
        let n_grid = r_grid.len();
        if n_grid == 0 {
            return Err(BarracudaError::invalid_input("r_grid cannot be empty"));
        }

        let batch_size = density.len() / n_grid;
        if density.len() != batch_size * n_grid {
            return Err(BarracudaError::invalid_input(
                "density length must be batch_size × n_grid",
            ));
        }

        let n_states = ls_factors.len() / batch_size;
        if ls_factors.len() != batch_size * n_states {
            return Err(BarracudaError::invalid_input(
                "ls_factors length must be batch_size × n_states",
            ));
        }

        if wf_squared.len() != batch_size * n_states * n_grid {
            return Err(BarracudaError::invalid_input(format!(
                "wf_squared length {} must be batch({}) × n_states({}) × n_grid({})",
                wf_squared.len(),
                batch_size,
                n_states,
                n_grid
            )));
        }

        self.compute_internal(&SpinOrbitInputs {
            wf_squared,
            drho_dr: None,
            density: Some(density),
            r_grid,
            ls_factors,
            batch_size,
            n_states,
            n_grid,
            dr,
            w0,
            entry_point: "spin_orbit_with_gradient",
        })
    }

    fn compute_internal(&self, inputs: &SpinOrbitInputs<'_>) -> Result<Vec<f64>> {
        let SpinOrbitInputs {
            wf_squared,
            drho_dr,
            density,
            r_grid,
            ls_factors,
            batch_size,
            n_states,
            n_grid,
            dr,
            w0,
            entry_point,
        } = inputs;
        // Create buffers
        let params = SpinOrbitParams::new(
            *batch_size as u32,
            *n_states as u32,
            *n_grid as u32,
            *dr,
            *w0,
        );
        let params_buffer = self
            .device
            .create_uniform_buffer("SpinOrbit Params", &params);

        let wf_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpinOrbit wf_squared"),
                contents: bytemuck::cast_slice(wf_squared),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let drho_owned = drho_dr.map(|drho| {
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SpinOrbit drho_dr"),
                    contents: bytemuck::cast_slice(drho),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        });
        let drho_buffer = drho_owned
            .as_ref()
            .unwrap_or_else(|| self.device.placeholder_buffer());

        let r_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpinOrbit r_grid"),
                contents: bytemuck::cast_slice(r_grid),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let ls_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpinOrbit ls_factors"),
                contents: bytemuck::cast_slice(ls_factors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (batch_size * n_states * 8) as u64;
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SpinOrbit h_so_diag"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let n_threads = batch_size * n_states;
        let n_workgroups = n_threads.div_ceil(WORKGROUP_SIZE_COMPACT as usize);

        if let Some(dens) = density {
            let density_buffer = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("SpinOrbit density"),
                    contents: bytemuck::cast_slice(dens),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            );
            ComputeDispatch::new(self.device.as_ref(), "SpinOrbit")
                .shader(Self::wgsl_shader(), entry_point)
                .f64()
                .uniform(0, &params_buffer)
                .storage_read(1, &wf_buffer)
                .storage_read(2, drho_buffer)
                .storage_read(3, &r_buffer)
                .storage_read(4, &ls_buffer)
                .storage_rw(5, &output_buffer)
                .storage_read(6, &density_buffer)
                .dispatch(n_workgroups as u32, 1, 1)
                .submit()?;
        } else {
            ComputeDispatch::new(self.device.as_ref(), "SpinOrbit")
                .shader(Self::wgsl_shader(), entry_point)
                .f64()
                .uniform(0, &params_buffer)
                .storage_read(1, &wf_buffer)
                .storage_read(2, drho_buffer)
                .storage_read(3, &r_buffer)
                .storage_read(4, &ls_buffer)
                .storage_rw(5, &output_buffer)
                .dispatch(n_workgroups as u32, 1, 1)
                .submit()?;
        }

        // Read back results
        self.device
            .read_f64_buffer(&output_buffer, batch_size * n_states)
    }
}

/// Compute `ls_i` factor for a state with quantum numbers (l, j)
///
/// Formula: `ls_i` = (j(j+1) - l(l+1) - 3/4) / 2
///
/// # Arguments
/// * `l` - Orbital angular momentum quantum number
/// * `j` - Total angular momentum quantum number (l±1/2)
///
/// # Example
/// ```rust,ignore
/// let ls = compute_ls_factor(1, 1.5);  // p3/2 state
/// let ls = compute_ls_factor(1, 0.5);  // p1/2 state
/// ```
#[must_use]
pub fn compute_ls_factor(l: u32, j: f64) -> f64 {
    let l_f = l as f64;
    (j.mul_add(j + 1.0, -(l_f * (l_f + 1.0))) - 0.75) / 2.0
}

#[cfg(test)]
#[path = "spin_orbit_f64_tests.rs"]
mod tests;
