// SPDX-License-Identifier: AGPL-3.0-or-later

//! Axially-deformed HFB pipeline: BCS, density, energy, Hamiltonian, potentials, wavefunctions.
//!
//! Absorbed from hotSpring v0.6.4 handoff (Feb 2026).
//!
//! f64 shaders are compiled via `device.compile_shader_f64()` when Springs orchestrate dispatch.

use crate::device::WgpuDevice;
use std::sync::Arc;

/// WGSL source for deformed BCS occupations and particle number.
pub const WGSL_DEFORMED_BCS: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_bcs_f64.wgsl");
/// WGSL source for deformed density computation.
pub const WGSL_DEFORMED_DENSITY: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_density_f64.wgsl");
/// WGSL source for deformed energy integrand and reduction.
pub const WGSL_DEFORMED_ENERGY: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_energy_f64.wgsl");
/// WGSL source for deformed Hamiltonian construction.
pub const WGSL_DEFORMED_HAMILTONIAN: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_hamiltonian_f64.wgsl");
/// WGSL source for deformed potential computation.
pub const WGSL_DEFORMED_POTENTIAL: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_potential_f64.wgsl");
/// WGSL source for deformed wavefunction evaluation.
pub const WGSL_DEFORMED_WAVEFUNCTION: &str =
    include_str!("../../shaders/science/hfb_deformed/deformed_wavefunction_f64.wgsl");

/// Unified axially-deformed HFB pipeline: holds device reference and exposes shader constants.
///
/// Springs orchestrate multi-pass workflows using these constants and
/// `device.compile_shader_f64()` for f64 shader compilation.
pub struct DeformedHfbPipeline {
    device: Arc<WgpuDevice>,
}

impl DeformedHfbPipeline {
    /// Create a deformed HFB pipeline for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Reference to the underlying wgpu device.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deformed_bcs_non_empty_and_has_entries() {
        assert!(!WGSL_DEFORMED_BCS.is_empty());
        assert!(WGSL_DEFORMED_BCS.contains("fn bcs_occupations"));
        assert!(WGSL_DEFORMED_BCS.contains("fn particle_number_and_gap"));
    }

    #[test]
    fn deformed_density_non_empty_and_has_entries() {
        assert!(!WGSL_DEFORMED_DENSITY.is_empty());
        assert!(WGSL_DEFORMED_DENSITY.contains("fn compute_density"));
        assert!(WGSL_DEFORMED_DENSITY.contains("fn mix_density"));
    }

    #[test]
    fn deformed_energy_non_empty_and_has_entries() {
        assert!(!WGSL_DEFORMED_ENERGY.is_empty());
        assert!(WGSL_DEFORMED_ENERGY.contains("fn energy_integrand"));
        assert!(WGSL_DEFORMED_ENERGY.contains("fn reduce_energy"));
    }

    #[test]
    fn deformed_hamiltonian_non_empty_and_has_entry() {
        assert!(!WGSL_DEFORMED_HAMILTONIAN.is_empty());
        assert!(WGSL_DEFORMED_HAMILTONIAN.contains("fn build_hamiltonian"));
    }

    #[test]
    fn deformed_potential_non_empty_and_has_entry() {
        assert!(!WGSL_DEFORMED_POTENTIAL.is_empty());
        assert!(WGSL_DEFORMED_POTENTIAL.contains("fn compute_potentials"));
    }

    #[test]
    fn deformed_wavefunction_non_empty_and_has_entry() {
        assert!(!WGSL_DEFORMED_WAVEFUNCTION.is_empty());
        assert!(WGSL_DEFORMED_WAVEFUNCTION.contains("fn evaluate_basis"));
    }
}
