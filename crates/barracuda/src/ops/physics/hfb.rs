// SPDX-License-Identifier: AGPL-3.0-or-later

//! Spherical HFB pipeline: BCS bisection, density, energy, Hamiltonian, potentials.
//!
//! Absorbed from hotSpring v0.6.4 handoff (Feb 2026).
//!
//! f64 shaders are compiled via `device.compile_shader_f64()` when Springs orchestrate dispatch.

use crate::device::WgpuDevice;
use std::sync::Arc;

pub const WGSL_BCS_BISECTION: &str =
    include_str!("../../shaders/science/hfb/bcs_bisection_f64.wgsl");
pub const WGSL_HFB_DENSITY: &str =
    include_str!("../../shaders/science/hfb/batched_hfb_density_f64.wgsl");
pub const WGSL_HFB_ENERGY: &str =
    include_str!("../../shaders/science/hfb/batched_hfb_energy_f64.wgsl");
pub const WGSL_HFB_HAMILTONIAN: &str =
    include_str!("../../shaders/science/hfb/batched_hfb_hamiltonian_f64.wgsl");
pub const WGSL_HFB_POTENTIALS: &str =
    include_str!("../../shaders/science/hfb/batched_hfb_potentials_f64.wgsl");

/// Unified spherical HFB pipeline: holds device reference and exposes shader constants.
///
/// Springs orchestrate multi-pass workflows using these constants and
/// `device.compile_shader_f64()` for f64 shader compilation.
pub struct HfbPipeline {
    device: Arc<WgpuDevice>,
}

impl HfbPipeline {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcs_bisection_non_empty_and_has_entry() {
        assert!(!WGSL_BCS_BISECTION.is_empty());
        assert!(WGSL_BCS_BISECTION.contains("fn bisect"));
    }

    #[test]
    fn hfb_density_non_empty_and_has_entries() {
        assert!(!WGSL_HFB_DENSITY.is_empty());
        assert!(WGSL_HFB_DENSITY.contains("fn bcs_occupations"));
        assert!(WGSL_HFB_DENSITY.contains("fn compute_density"));
        assert!(WGSL_HFB_DENSITY.contains("fn mix_density"));
    }

    #[test]
    fn hfb_energy_non_empty_and_has_entries() {
        assert!(!WGSL_HFB_ENERGY.is_empty());
        assert!(WGSL_HFB_ENERGY.contains("fn energy_integrands"));
        assert!(WGSL_HFB_ENERGY.contains("fn reduce_energy"));
    }

    #[test]
    fn hfb_hamiltonian_non_empty_and_has_entry() {
        assert!(!WGSL_HFB_HAMILTONIAN.is_empty());
        assert!(WGSL_HFB_HAMILTONIAN.contains("fn build_hamiltonian"));
    }

    #[test]
    fn hfb_potentials_non_empty_and_has_entry() {
        assert!(!WGSL_HFB_POTENTIALS.is_empty());
        assert!(WGSL_HFB_POTENTIALS.contains("fn compute_potentials"));
    }
}
