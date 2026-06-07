// SPDX-License-Identifier: AGPL-3.0-or-later
//! Physics WGSL shaders absorbed from hotSpring (Jun 2026).
//!
//! Consolidates all 19 hotSpring `physics/shaders/` sources. Spherical and
//! deformed HFB shaders live under `shaders/science/hfb*`; plasma shaders under
//! `shaders/science/plasma/`; newly absorbed shaders under `shaders/physics/`.

// ── Spherical HFB (5 + spin-orbit pack) ────────────────────────────────

pub use super::hfb::{
    WGSL_BCS_BISECTION, WGSL_HFB_DENSITY, WGSL_HFB_ENERGY, WGSL_HFB_HAMILTONIAN,
    WGSL_HFB_POTENTIALS, WGSL_SPIN_ORBIT_PACK,
};

// ── Axially-deformed HFB (6 + density/energy + gradient) ────────────────

pub use super::hfb_deformed::{
    WGSL_DEFORMED_BCS, WGSL_DEFORMED_DENSITY, WGSL_DEFORMED_DENSITY_ENERGY, WGSL_DEFORMED_ENERGY,
    WGSL_DEFORMED_GRADIENT, WGSL_DEFORMED_HAMILTONIAN, WGSL_DEFORMED_POTENTIAL,
    WGSL_DEFORMED_WAVEFUNCTION,
};

// ── SEMF / nuclear EOS L1 (3) ───────────────────────────────────────────

pub use super::semf::{WGSL_CHI2_BATCH, WGSL_SEMF_BATCH, WGSL_SEMF_PURE_GPU};

// ── Kinetic plasma / dielectric (4) ───────────────────────────────────

pub use super::plasma::{
    WGSL_BGK_RELAXATION, WGSL_DIELECTRIC_MERMIN, WGSL_DIELECTRIC_MULTICOMPONENT, WGSL_EULER_HLL,
};

// ── Metadynamics FES (1) ────────────────────────────────────────────────

pub use super::fes::WGSL_FES_GAUSSIAN_SUM;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_non_empty(name: &str, src: &str) {
        assert!(
            src.len() > 50,
            "{name} shader source is too short ({} bytes)",
            src.len()
        );
    }

    fn assert_agpl_header(name: &str, src: &str) {
        assert!(
            src.starts_with("// SPDX-License-Identifier: AGPL-3.0"),
            "{name} must start with AGPL SPDX header"
        );
    }

    #[test]
    fn all_nineteen_physics_shaders_non_empty() {
        for (name, src) in [
            ("bcs_bisection", WGSL_BCS_BISECTION),
            ("hfb_density", WGSL_HFB_DENSITY),
            ("hfb_energy", WGSL_HFB_ENERGY),
            ("hfb_hamiltonian", WGSL_HFB_HAMILTONIAN),
            ("hfb_potentials", WGSL_HFB_POTENTIALS),
            ("spin_orbit_pack", WGSL_SPIN_ORBIT_PACK),
            ("deformed_bcs", WGSL_DEFORMED_BCS),
            ("deformed_density", WGSL_DEFORMED_DENSITY),
            ("deformed_density_energy", WGSL_DEFORMED_DENSITY_ENERGY),
            ("deformed_energy", WGSL_DEFORMED_ENERGY),
            ("deformed_gradient", WGSL_DEFORMED_GRADIENT),
            ("deformed_hamiltonian", WGSL_DEFORMED_HAMILTONIAN),
            ("deformed_potential", WGSL_DEFORMED_POTENTIAL),
            ("deformed_wavefunction", WGSL_DEFORMED_WAVEFUNCTION),
            ("semf_batch", WGSL_SEMF_BATCH),
            ("semf_pure_gpu", WGSL_SEMF_PURE_GPU),
            ("chi2_batch", WGSL_CHI2_BATCH),
            ("bgk_relaxation", WGSL_BGK_RELAXATION),
            ("euler_hll", WGSL_EULER_HLL),
            ("dielectric_mermin", WGSL_DIELECTRIC_MERMIN),
            ("dielectric_multicomponent", WGSL_DIELECTRIC_MULTICOMPONENT),
            ("fes_gaussian_sum", WGSL_FES_GAUSSIAN_SUM),
        ] {
            assert_non_empty(name, src);
            assert_agpl_header(name, src);
        }
    }
}
