// SPDX-License-Identifier: AGPL-3.0-or-later
//! MD WGSL shaders absorbed from hotSpring (Jun 2026).
//!
//! Production GPU simulation shaders for Yukawa OCP, split Velocity-Verlet,
//! Berendsen thermostat, cell-list / Verlet forces, transport observables,
//! and Verlet neighbor-list maintenance.
//!
//! Canonical WGSL sources live in [`../../shaders/md/`](../../shaders/md/).
//! ESN readout is re-exported from [`crate::esn_v2`] (already upstream in `ml/`).

// ── Yukawa force kernels ─────────────────────────────────────────────

/// Yukawa all-pairs force (f64) with PBC minimum-image and PE accumulation.
pub const WGSL_YUKAWA_FORCE_F64: &str = include_str!("../../shaders/md/yukawa_force_f64.wgsl");

/// Yukawa cell-list force (f64) — 27-cell stencil.
pub const WGSL_YUKAWA_FORCE_CELLLIST_F64: &str =
    include_str!("../../shaders/md/yukawa_force_celllist_f64.wgsl");

/// Yukawa cell-list force v2 (f64) — flat neighbor loop.
pub const WGSL_YUKAWA_FORCE_CELLLIST_V2_F64: &str =
    include_str!("../../shaders/md/yukawa_force_celllist_v2_f64.wgsl");

/// Yukawa cell-list force with indirect indexing (f64).
pub const WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_F64: &str =
    include_str!("../../shaders/md/yukawa_force_celllist_indirect_f64.wgsl");

/// Yukawa all-pairs force (DF64) — FP32-core streaming variant.
pub const WGSL_YUKAWA_FORCE_DF64: &str = include_str!("../../shaders/md/yukawa_force_df64.wgsl");

/// Yukawa cell-list indirect force (DF64).
pub const WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_DF64: &str =
    include_str!("../../shaders/md/yukawa_force_celllist_indirect_df64.wgsl");

/// Yukawa Verlet neighbor-list force (f64).
pub const WGSL_YUKAWA_FORCE_VERLET_F64: &str =
    include_str!("../../shaders/md/yukawa_force_verlet_f64.wgsl");

/// Yukawa Verlet neighbor-list force (DF64).
pub const WGSL_YUKAWA_FORCE_VERLET_DF64: &str =
    include_str!("../../shaders/md/yukawa_force_verlet_df64.wgsl");

// ── Integrators & thermostats ────────────────────────────────────────

/// Split Velocity-Verlet kick-drift + PBC wrap (f64).
pub const WGSL_VV_KICK_DRIFT_F64: &str = include_str!("../../shaders/md/vv_kick_drift_f64.wgsl");

/// Split Velocity-Verlet second half-kick (f64).
pub const WGSL_VV_HALF_KICK_F64: &str = include_str!("../../shaders/md/vv_half_kick_f64.wgsl");

/// Berendsen velocity-rescaling thermostat (f64).
pub const WGSL_BERENDSEN_F64: &str = include_str!("../../shaders/md/berendsen_f64.wgsl");

// ── Observables ──────────────────────────────────────────────────────

/// Per-particle kinetic energy reduction (f64).
pub const WGSL_KINETIC_ENERGY_F64: &str = include_str!("../../shaders/md/kinetic_energy_f64.wgsl");

/// RDF histogram binning (f64).
pub const WGSL_RDF_HISTOGRAM_F64: &str = include_str!("../../shaders/md/rdf_histogram_f64.wgsl");

/// Batched VACF — one dispatch computes C(lag) across all origins (f64).
pub const WGSL_VACF_BATCH_F64: &str = include_str!("../../shaders/md/vacf_batch_f64.wgsl");

/// Per-particle v(t0) · v(t) dot product for VACF (f64).
pub const WGSL_VACF_DOT_F64: &str = include_str!("../../shaders/md/vacf_dot_f64.wgsl");

/// Per-particle stress virial σ_xy for Green-Kubo viscosity (f64).
pub const WGSL_STRESS_VIRIAL_F64: &str = include_str!("../../shaders/md/stress_virial_f64.wgsl");

// ── Verlet neighbor list ─────────────────────────────────────────────

/// Build Verlet neighbor list from cell-list stencil.
pub const WGSL_VERLET_BUILD: &str = include_str!("../../shaders/md/verlet_build.wgsl");

/// Check max displacement against skin/2 rebuild threshold.
pub const WGSL_VERLET_CHECK_DISPLACEMENT: &str =
    include_str!("../../shaders/md/verlet_check_displacement.wgsl");

/// Copy current positions to reference buffer after rebuild.
pub const WGSL_VERLET_COPY_REF: &str = include_str!("../../shaders/md/verlet_copy_ref.wgsl");

// ── ESN (Echo State Network) ─────────────────────────────────────────

/// ESN reservoir update (f32) — fused W_in·input + W_res·state → leaky tanh.
pub const WGSL_ESN_RESERVOIR_UPDATE: &str =
    include_str!("../../shaders/md/esn_reservoir_update.wgsl");

/// ESN readout (f32) — re-exported from upstream `esn_v2` (`ml/esn_readout.wgsl`).
pub use crate::esn_v2::WGSL_ESN_READOUT;

// ── hotSpring compatibility aliases (SHADER_* naming) ────────────────

/// Alias for [`WGSL_YUKAWA_FORCE_F64`].
pub const SHADER_YUKAWA_FORCE: &str = WGSL_YUKAWA_FORCE_F64;
/// Alias for [`WGSL_VV_KICK_DRIFT_F64`].
pub const SHADER_VV_KICK_DRIFT: &str = WGSL_VV_KICK_DRIFT_F64;
/// Alias for [`WGSL_VV_HALF_KICK_F64`].
pub const SHADER_VV_HALF_KICK: &str = WGSL_VV_HALF_KICK_F64;
/// Alias for [`WGSL_BERENDSEN_F64`].
pub const SHADER_BERENDSEN: &str = WGSL_BERENDSEN_F64;
/// Alias for [`WGSL_KINETIC_ENERGY_F64`].
pub const SHADER_KINETIC_ENERGY: &str = WGSL_KINETIC_ENERGY_F64;
/// Alias for [`WGSL_YUKAWA_FORCE_CELLLIST_F64`].
pub const SHADER_YUKAWA_FORCE_CELLLIST: &str = WGSL_YUKAWA_FORCE_CELLLIST_F64;
/// Alias for [`WGSL_YUKAWA_FORCE_CELLLIST_V2_F64`].
pub const SHADER_YUKAWA_FORCE_CELLLIST_V2: &str = WGSL_YUKAWA_FORCE_CELLLIST_V2_F64;
/// Alias for [`WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_F64`].
pub const SHADER_YUKAWA_FORCE_INDIRECT: &str = WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_F64;
/// Alias for [`WGSL_YUKAWA_FORCE_DF64`].
pub const SHADER_YUKAWA_FORCE_DF64: &str = WGSL_YUKAWA_FORCE_DF64;
/// Alias for [`WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_DF64`].
pub const SHADER_YUKAWA_FORCE_INDIRECT_DF64: &str = WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_DF64;
/// Alias for [`WGSL_YUKAWA_FORCE_VERLET_F64`].
pub const SHADER_YUKAWA_FORCE_VERLET: &str = WGSL_YUKAWA_FORCE_VERLET_F64;
/// Alias for [`WGSL_YUKAWA_FORCE_VERLET_DF64`].
pub const SHADER_YUKAWA_FORCE_VERLET_DF64: &str = WGSL_YUKAWA_FORCE_VERLET_DF64;
/// Alias for [`WGSL_RDF_HISTOGRAM_F64`].
pub const SHADER_RDF_HISTOGRAM: &str = WGSL_RDF_HISTOGRAM_F64;
/// Alias for [`WGSL_ESN_RESERVOIR_UPDATE`].
pub const SHADER_ESN_RESERVOIR_UPDATE: &str = WGSL_ESN_RESERVOIR_UPDATE;
/// Alias for [`WGSL_ESN_READOUT`].
pub const SHADER_ESN_READOUT: &str = WGSL_ESN_READOUT;
/// Alias for [`WGSL_VERLET_BUILD`].
pub const SHADER_VERLET_BUILD: &str = WGSL_VERLET_BUILD;
/// Alias for [`WGSL_VERLET_CHECK_DISPLACEMENT`].
pub const SHADER_VERLET_CHECK_DISP: &str = WGSL_VERLET_CHECK_DISPLACEMENT;
/// Alias for [`WGSL_VERLET_COPY_REF`].
pub const SHADER_VERLET_COPY_REF: &str = WGSL_VERLET_COPY_REF;

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

    fn assert_has_compute(name: &str, src: &str) {
        assert!(src.contains("@compute"), "{name} must contain @compute");
        assert!(
            src.contains("@workgroup_size"),
            "{name} must contain @workgroup_size"
        );
    }

    fn assert_agpl_header(name: &str, src: &str) {
        assert!(
            src.starts_with("// SPDX-License-Identifier: AGPL-3.0"),
            "{name} must start with AGPL SPDX header"
        );
    }

    #[test]
    fn all_absorbed_shaders_valid() {
        for (name, src) in [
            ("yukawa_force_f64", WGSL_YUKAWA_FORCE_F64),
            ("yukawa_force_celllist_f64", WGSL_YUKAWA_FORCE_CELLLIST_F64),
            ("yukawa_force_celllist_v2_f64", WGSL_YUKAWA_FORCE_CELLLIST_V2_F64),
            (
                "yukawa_force_celllist_indirect_f64",
                WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_F64,
            ),
            ("yukawa_force_df64", WGSL_YUKAWA_FORCE_DF64),
            (
                "yukawa_force_celllist_indirect_df64",
                WGSL_YUKAWA_FORCE_CELLLIST_INDIRECT_DF64,
            ),
            ("yukawa_force_verlet_f64", WGSL_YUKAWA_FORCE_VERLET_F64),
            ("yukawa_force_verlet_df64", WGSL_YUKAWA_FORCE_VERLET_DF64),
            ("vv_kick_drift_f64", WGSL_VV_KICK_DRIFT_F64),
            ("vv_half_kick_f64", WGSL_VV_HALF_KICK_F64),
            ("berendsen_f64", WGSL_BERENDSEN_F64),
            ("kinetic_energy_f64", WGSL_KINETIC_ENERGY_F64),
            ("rdf_histogram_f64", WGSL_RDF_HISTOGRAM_F64),
            ("vacf_batch_f64", WGSL_VACF_BATCH_F64),
            ("vacf_dot_f64", WGSL_VACF_DOT_F64),
            ("stress_virial_f64", WGSL_STRESS_VIRIAL_F64),
            ("verlet_build", WGSL_VERLET_BUILD),
            ("verlet_check_displacement", WGSL_VERLET_CHECK_DISPLACEMENT),
            ("verlet_copy_ref", WGSL_VERLET_COPY_REF),
            ("esn_reservoir_update", WGSL_ESN_RESERVOIR_UPDATE),
            ("esn_readout", WGSL_ESN_READOUT),
        ] {
            assert_non_empty(name, src);
            assert_has_compute(name, src);
            assert_agpl_header(name, src);
        }
    }

    #[test]
    fn hotspring_aliases_match_wgsl_sources() {
        assert_eq!(SHADER_YUKAWA_FORCE, WGSL_YUKAWA_FORCE_F64);
        assert_eq!(SHADER_VV_KICK_DRIFT, WGSL_VV_KICK_DRIFT_F64);
        assert_eq!(SHADER_ESN_READOUT, WGSL_ESN_READOUT);
    }
}
