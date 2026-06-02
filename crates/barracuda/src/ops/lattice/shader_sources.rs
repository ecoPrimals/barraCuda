// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fully merged WGSL sources for the lattice QCD shader corpus.
//!
//! [`lattice_shader_source`] prepends the correct math preamble chain so each
//! kernel can compile standalone (e.g. through coralReef IPC). Library shaders
//! (`complex_f64`, `prng_pcg_f64`, `su3_math_f64`, `su3_f64`) return their raw
//! body with no preamble.

use super::absorbed_shaders::{
    WGSL_AXPY_FILE_F64, WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64, WGSL_CG_UPDATE_XR_SHIFTED_F64,
    WGSL_COMPLEX_DOT_RE_FILE_F64, WGSL_FERMION_ACTION_SUM_F64, WGSL_GAUSSIAN_FERMION_F64, WGSL_GPU_METROPOLIS_F64,
    WGSL_HAMILTONIAN_ASSEMBLY_F64, WGSL_MS_P_UPDATE_F64, WGSL_MS_X_UPDATE_F64,
    WGSL_MS_ZETA_UPDATE_F64, WGSL_PRNG_PCG_F64, WGSL_STAGGERED_FERMION_FORCE_F64,
    WGSL_SU3_FERMION_FORCE_ACCUMULATE_ROP_F64, WGSL_SU3_FORCE_ATOMIC_TO_MOMENTUM_F64,
    WGSL_SU3_GAUGE_FORCE_DF64, WGSL_SU3_GAUGE_FORCE_F64, WGSL_SU3_KINETIC_ENERGY_DF64,
    WGSL_SU3_KINETIC_ENERGY_F64, WGSL_SU3_LATTICE_F64, WGSL_SU3_LINK_UPDATE_F64,
    WGSL_SU3_MATH_F64, WGSL_SU3_MOMENTUM_UPDATE_F64, WGSL_SU3_RANDOM_MOMENTA_F64,
    WGSL_SU3_RANDOM_MOMENTA_TMU_F64, WGSL_SUM_REDUCE_SUBGROUP_F64, WGSL_XPAY_FILE_F64,
};
use super::cg::{
    WGSL_CG_COMPUTE_ALPHA_F64, WGSL_CG_COMPUTE_BETA_F64, WGSL_CG_KERNELS_F64,
    WGSL_CG_UPDATE_P_F64, WGSL_CG_UPDATE_XR_F64, WGSL_SUM_REDUCE_F64,
};
use super::complex_f64::WGSL_COMPLEX64;
use super::lcg::WGSL_LCG_F64;
use super::su3::{su3_df64_preamble, su3_preamble, WGSL_SU3};
use super::su3_extended::{su3_extended_preamble, WGSL_SU3_EXTENDED};

const WGSL_WILSON_PLAQUETTE_F64: &str =
    include_str!("../../shaders/lattice/wilson_plaquette_f64.wgsl");
const WGSL_WILSON_PLAQUETTE_DF64: &str =
    include_str!("../../shaders/lattice/wilson_plaquette_df64.wgsl");
const WGSL_WILSON_ACTION_F64: &str = include_str!("../../shaders/lattice/wilson_action_f64.wgsl");
const WGSL_WILSON_ACTION_DF64: &str = include_str!("../../shaders/lattice/wilson_action_df64.wgsl");
const WGSL_SU3_HMC_FORCE_F64: &str = include_str!("../../shaders/lattice/su3_hmc_force_f64.wgsl");
const WGSL_SU3_HMC_FORCE_DF64: &str =
    include_str!("../../shaders/lattice/su3_hmc_force_df64.wgsl");
const WGSL_KINETIC_ENERGY_F64: &str = include_str!("../../shaders/lattice/kinetic_energy_f64.wgsl");
const WGSL_KINETIC_ENERGY_DF64: &str =
    include_str!("../../shaders/lattice/kinetic_energy_df64.wgsl");
const WGSL_HMC_LEAPFROG_F64: &str = include_str!("../../shaders/lattice/hmc_leapfrog_f64.wgsl");
const WGSL_LATTICE_INIT_F64: &str = include_str!("../../shaders/lattice/lattice_init_f64.wgsl");
const WGSL_PSEUDOFERMION_HEATBATH_F64: &str =
    include_str!("../../shaders/lattice/pseudofermion_heatbath_f64.wgsl");
const WGSL_PSEUDOFERMION_FORCE_F64: &str =
    include_str!("../../shaders/lattice/pseudofermion_force_f64.wgsl");
const WGSL_POLYAKOV_LOOP_F64: &str = include_str!("../../shaders/lattice/polyakov_loop_f64.wgsl");
const WGSL_HIGGS_U1_HMC_F64: &str = include_str!("../../shaders/lattice/higgs_u1_hmc_f64.wgsl");
const WGSL_DIRAC_STAGGERED_F64: &str =
    include_str!("../../shaders/lattice/dirac_staggered_f64.wgsl");
const WGSL_SU3_FLOW_ACCUMULATE_F64: &str =
    include_str!("../../shaders/lattice/su3_flow_accumulate_f64.wgsl");

/// Preamble requirement for a lattice kernel shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreambleKind {
    /// Kernel body is standalone (includes its own helpers/bindings).
    None,
    /// `complex_f64.wgsl` only (U(1) Higgs).
    Complex,
    /// `complex_f64` + `su3.wgsl`.
    Su3,
    /// `complex_f64` + `su3` + `lcg_f64` + `su3_extended_f64`.
    Su3Extended,
    /// `prng_pcg_f64.wgsl` (TMU momenta; body uses `uniform_f64` from preamble).
    PrngPcg,
    /// DF64 stack: `complex_f64` + `su3` + `df64_core` + `df64_transcendentals` + `su3_df64`.
    Df64,
}

/// Universal lattice preamble: `complex_f64` + `su3` + `lcg_f64` + `su3_extended_f64`.
///
/// Works for all shaders that depend on extended SU(3) operations (Cayley exp,
/// reunitarize, random SU(3)).
#[must_use]
pub fn lattice_preamble() -> String {
    format!(
        "{WGSL_COMPLEX64}\n{WGSL_SU3}\n{WGSL_LCG_F64}\n{WGSL_SU3_EXTENDED}\n"
    )
}

/// DF64 lattice preamble: `complex_f64` + `su3` + `df64_core` + `df64_transcendentals` + `su3_df64`.
#[must_use]
pub fn lattice_preamble_df64() -> String {
    su3_df64_preamble()
}

/// Canonical hotSpring lattice QCD shader corpus (51 kernels + libraries).
pub const LATTICE_SHADER_NAMES: &[&str] = &[
    "wilson_action_f64",
    "wilson_action_df64",
    "su3_hmc_force_f64",
    "su3_hmc_force_df64",
    "pseudofermion_heatbath_f64",
    "pseudofermion_force_f64",
    "multi_shift_zeta_f64",
    "lattice_init_f64",
    "kinetic_energy_f64",
    "kinetic_energy_df64",
    "hmc_leapfrog_f64",
    "higgs_u1_hmc_f64",
    "cg_kernels_f64",
    "wilson_plaquette_df64",
    "su3_kinetic_energy_df64",
    "wilson_plaquette_f64",
    "xpay_f64",
    "su3_random_momenta_f64",
    "dirac_staggered_f64",
    "fermion_action_sum_f64",
    "complex_f64",
    "sum_reduce_subgroup_f64",
    "su3_gauge_force_f64",
    "polyakov_loop_f64",
    "cg_update_xr_shifted_f64",
    "su3_f64",
    "cg_update_p_f64",
    "su3_force_atomic_to_momentum_f64",
    "cg_compute_beta_f64",
    "su3_math_f64",
    "cg_update_xr_f64",
    "hamiltonian_assembly_f64",
    "su3_momentum_update_f64",
    "complex_dot_re_f64",
    "gaussian_fermion_f64",
    "cg_compute_alpha_shifted_f64",
    "axpy_f64",
    "su3_flow_accumulate_f64",
    "cg_compute_alpha_f64",
    "ms_x_update_f64",
    "metropolis_f64",
    "ms_p_update_f64",
    "prng_pcg_f64",
    "su3_gauge_force_df64",
    "staggered_fermion_force_f64",
    "su3_fermion_force_accumulate_rop_f64",
    "su3_random_momenta_tmu_f64",
    "ms_zeta_update_f64",
    "su3_link_update_f64",
    "su3_kinetic_energy_f64",
    "sum_reduce_f64",
];

/// All registered lattice shader names.
#[must_use]
pub fn lattice_shader_names() -> &'static [&'static str] {
    LATTICE_SHADER_NAMES
}

fn merge_shader(preamble: PreambleKind, body: &str) -> String {
    match preamble {
        PreambleKind::None => body.to_string(),
        PreambleKind::Complex => format!("{WGSL_COMPLEX64}\n{body}"),
        PreambleKind::Su3 => format!("{}{}", su3_preamble(), body),
        PreambleKind::Su3Extended => format!("{}{}", su3_extended_preamble(), body),
        PreambleKind::PrngPcg => format!("{WGSL_PRNG_PCG_F64}\n{body}"),
        PreambleKind::Df64 => format!("{}{}", lattice_preamble_df64(), body),
    }
}

fn lookup_shader(name: &str) -> Option<(&'static str, PreambleKind)> {
    Some(match name {
        // ── Self-contained kernels (no external preamble) ─────────────────
        "su3_gauge_force_f64" => (WGSL_SU3_GAUGE_FORCE_F64, PreambleKind::None),
        "su3_link_update_f64" => (WGSL_SU3_LINK_UPDATE_F64, PreambleKind::None),
        "su3_momentum_update_f64" => (WGSL_SU3_MOMENTUM_UPDATE_F64, PreambleKind::None),
        "su3_kinetic_energy_f64" => (WGSL_SU3_KINETIC_ENERGY_F64, PreambleKind::None),
        "dirac_staggered_f64" => (WGSL_DIRAC_STAGGERED_F64, PreambleKind::None),
        "staggered_fermion_force_f64" => (WGSL_STAGGERED_FERMION_FORCE_F64, PreambleKind::None),
        "cg_kernels_f64" => (WGSL_CG_KERNELS_F64, PreambleKind::None),
        "axpy_f64" => (WGSL_AXPY_FILE_F64, PreambleKind::None),
        "xpay_f64" => (WGSL_XPAY_FILE_F64, PreambleKind::None),
        "complex_dot_re_f64" => (WGSL_COMPLEX_DOT_RE_FILE_F64, PreambleKind::None),
        "sum_reduce_f64" => (WGSL_SUM_REDUCE_F64, PreambleKind::None),
        "sum_reduce_subgroup_f64" => (WGSL_SUM_REDUCE_SUBGROUP_F64, PreambleKind::None),
        "hamiltonian_assembly_f64" => (WGSL_HAMILTONIAN_ASSEMBLY_F64, PreambleKind::None),
        "fermion_action_sum_f64" => (WGSL_FERMION_ACTION_SUM_F64, PreambleKind::None),
        "metropolis_f64" | "gpu_metropolis_f64" => (WGSL_GPU_METROPOLIS_F64, PreambleKind::None),
        "gaussian_fermion_f64" => (WGSL_GAUSSIAN_FERMION_F64, PreambleKind::None),
        "ms_zeta_update_f64" | "multi_shift_zeta_f64" => {
            (WGSL_MS_ZETA_UPDATE_F64, PreambleKind::None)
        }
        "ms_x_update_f64" => (WGSL_MS_X_UPDATE_F64, PreambleKind::None),
        "ms_p_update_f64" => (WGSL_MS_P_UPDATE_F64, PreambleKind::None),
        "cg_compute_alpha_shifted_f64" => (WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64, PreambleKind::None),
        "cg_update_xr_shifted_f64" => (WGSL_CG_UPDATE_XR_SHIFTED_F64, PreambleKind::None),
        "cg_compute_alpha_f64" => (WGSL_CG_COMPUTE_ALPHA_F64, PreambleKind::None),
        "cg_compute_beta_f64" => (WGSL_CG_COMPUTE_BETA_F64, PreambleKind::None),
        "cg_update_xr_f64" => (WGSL_CG_UPDATE_XR_F64, PreambleKind::None),
        "cg_update_p_f64" => (WGSL_CG_UPDATE_P_F64, PreambleKind::None),
        "su3_fermion_force_accumulate_rop_f64" => {
            (WGSL_SU3_FERMION_FORCE_ACCUMULATE_ROP_F64, PreambleKind::None)
        }
        "su3_force_atomic_to_momentum_f64" => {
            (WGSL_SU3_FORCE_ATOMIC_TO_MOMENTUM_F64, PreambleKind::None)
        }
        "su3_flow_accumulate_f64" => (WGSL_SU3_FLOW_ACCUMULATE_F64, PreambleKind::None),
        // Inline PCG — prepending `prng_pcg_f64` would duplicate `pcg_hash`.
        "su3_random_momenta_f64" => (WGSL_SU3_RANDOM_MOMENTA_F64, PreambleKind::None),

        // ── Math / PRNG library shaders (raw body only) ─────────────────
        "complex_f64" => (WGSL_COMPLEX64, PreambleKind::None),
        "prng_pcg_f64" => (WGSL_PRNG_PCG_F64, PreambleKind::None),
        "su3_math_f64" => (WGSL_SU3_MATH_F64, PreambleKind::None),
        "su3_f64" => (WGSL_SU3_LATTICE_F64, PreambleKind::None),

        // ── `su3_preamble` (complex + su3) ──────────────────────────────
        "wilson_action_f64" => (WGSL_WILSON_ACTION_F64, PreambleKind::Su3),
        "su3_hmc_force_f64" => (WGSL_SU3_HMC_FORCE_F64, PreambleKind::Su3),
        "pseudofermion_force_f64" => (WGSL_PSEUDOFERMION_FORCE_F64, PreambleKind::Su3),
        "kinetic_energy_f64" => (WGSL_KINETIC_ENERGY_F64, PreambleKind::Su3),
        "polyakov_loop_f64" => (WGSL_POLYAKOV_LOOP_F64, PreambleKind::Su3),
        // Wilson plaquette uses `c64_*` / `su3_*` helpers from the shared libraries.
        "wilson_plaquette_f64" => (WGSL_WILSON_PLAQUETTE_F64, PreambleKind::Su3),

        // ── `su3_extended_preamble` (complex + su3 + lcg + su3_extended) ─
        "hmc_leapfrog_f64" => (WGSL_HMC_LEAPFROG_F64, PreambleKind::Su3Extended),
        "lattice_init_f64" => (WGSL_LATTICE_INIT_F64, PreambleKind::Su3Extended),
        "pseudofermion_heatbath_f64" => (WGSL_PSEUDOFERMION_HEATBATH_F64, PreambleKind::Su3Extended),

        // ── U(1) Higgs: complex arithmetic only ───────────────────────────
        "higgs_u1_hmc_f64" => (WGSL_HIGGS_U1_HMC_F64, PreambleKind::Complex),

        // ── PCG preamble prepend ──────────────────────────────────────────
        "su3_random_momenta_tmu_f64" => (WGSL_SU3_RANDOM_MOMENTA_TMU_F64, PreambleKind::PrngPcg),

        // ── DF64 preamble ─────────────────────────────────────────────────
        "wilson_plaquette_df64" => (WGSL_WILSON_PLAQUETTE_DF64, PreambleKind::Df64),
        "wilson_action_df64" => (WGSL_WILSON_ACTION_DF64, PreambleKind::Df64),
        "su3_gauge_force_df64" => (WGSL_SU3_GAUGE_FORCE_DF64, PreambleKind::Df64),
        "su3_hmc_force_df64" => (WGSL_SU3_HMC_FORCE_DF64, PreambleKind::Df64),
        "su3_kinetic_energy_df64" => (WGSL_SU3_KINETIC_ENERGY_DF64, PreambleKind::Df64),
        "kinetic_energy_df64" => (WGSL_KINETIC_ENERGY_DF64, PreambleKind::Df64),

        _ => return None,
    })
}

/// Return fully merged WGSL for a lattice shader by name.
///
/// Kernel shaders receive the appropriate preamble chain; library shaders and
/// self-contained kernels return the raw body. Returns `None` for unknown names.
#[must_use]
pub fn lattice_shader_source(name: &str) -> Option<String> {
    let (body, preamble) = lookup_shader(name)?;
    Some(merge_shader(preamble, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_preamble_matches_extended_stack() {
        assert_eq!(lattice_preamble(), su3_extended_preamble());
    }

    #[test]
    fn lattice_preamble_df64_matches_su3_df64_stack() {
        assert_eq!(lattice_preamble_df64(), su3_df64_preamble());
    }

    #[test]
    fn all_lattice_shaders_retrievable() {
        assert_eq!(LATTICE_SHADER_NAMES.len(), 51);
        for name in LATTICE_SHADER_NAMES {
            let src = lattice_shader_source(name)
                .unwrap_or_else(|| panic!("lattice_shader_source({name:?}) returned None"));
            assert!(!src.is_empty(), "{name} merged source is empty");
            assert!(
                src.len() > 50,
                "{name} merged source too short ({} bytes)",
                src.len()
            );
        }
    }

    #[test]
    fn su3_preamble_shaders_include_complex_and_su3() {
        let src = lattice_shader_source("wilson_action_f64").expect("wilson_action_f64");
        assert!(src.contains("c64_mul"), "must include complex_f64 preamble");
        assert!(src.contains("su3_plaquette"), "must include su3 preamble");
    }

    #[test]
    fn df64_shaders_include_df64_core() {
        let src = lattice_shader_source("wilson_plaquette_df64").expect("wilson_plaquette_df64");
        assert!(src.contains("df64_add"), "must include df64_core preamble");
    }

    #[test]
    fn tmu_momenta_includes_pcg_preamble() {
        let src =
            lattice_shader_source("su3_random_momenta_tmu_f64").expect("su3_random_momenta_tmu_f64");
        assert!(src.contains("uniform_f64"), "must prepend prng_pcg_f64");
    }
}
