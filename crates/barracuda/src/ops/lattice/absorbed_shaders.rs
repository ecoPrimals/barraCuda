// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lattice shaders absorbed from hotSpring (Feb 2026).
//!
//! These WGSL shaders are self-contained and available for GPU-resident
//! lattice QCD operations. Organized by absorption wave:
//!
//! - **S60**: Core HMC shaders (gauge force, kinetic energy, link/momentum update)
//! - **S64**: SU(3) math, PCG PRNG, DF64 gauge force + kinetic energy

// Re-export GPU-resident CG shaders from cg module
pub use super::cg::{
    WGSL_CG_COMPUTE_ALPHA_F64, WGSL_CG_COMPUTE_BETA_F64, WGSL_CG_UPDATE_P_F64,
    WGSL_CG_UPDATE_XR_F64, WGSL_SUM_REDUCE_F64,
};

// ── S60 absorption: core HMC ──────────────────────────────────────────

/// SU(3) gauge force: staple sum + traceless anti-Hermitian projection.
pub const WGSL_SU3_GAUGE_FORCE_F64: &str =
    include_str!("../../shaders/lattice/su3_gauge_force_f64.wgsl");

/// Kinetic energy: `T_link` = -0.5 * Re Tr(P²) per link.
pub const WGSL_SU3_KINETIC_ENERGY_F64: &str =
    include_str!("../../shaders/lattice/su3_kinetic_energy_f64.wgsl");

/// Link update: U = exp(dt * P) * U via Cayley + reunitarize.
pub const WGSL_SU3_LINK_UPDATE_F64: &str =
    include_str!("../../shaders/lattice/su3_link_update_f64.wgsl");

/// Momentum update: P += dt * F.
pub const WGSL_SU3_MOMENTUM_UPDATE_F64: &str =
    include_str!("../../shaders/lattice/su3_momentum_update_f64.wgsl");

/// SU(3) algebra momentum generation via PCG hash PRNG.
pub const WGSL_SU3_RANDOM_MOMENTA_F64: &str =
    include_str!("../../shaders/lattice/su3_random_momenta_f64.wgsl");

/// Gaussian random fermion field (η ~ N(0,1)) for pseudofermion heat bath.
pub const WGSL_GAUSSIAN_FERMION_F64: &str =
    include_str!("../../shaders/lattice/gaussian_fermion_f64.wgsl");

/// Staggered fermion force: `F_f` = TA[ U·M ].
pub const WGSL_STAGGERED_FERMION_FORCE_F64: &str =
    include_str!("../../shaders/lattice/staggered_fermion_force_f64.wgsl");

// ── S64 absorption: SU(3) math, PRNG, DF64 ───────────────────────────

/// Naga-safe SU(3) pure math library (composition-safe, no I/O bindings).
///
/// Fixes naga composition bug in the original `su3_f64.wgsl` by separating
/// math functions from storage/uniform declarations. Prepend to any shader
/// that needs SU(3) algebra.
pub const WGSL_SU3_MATH_F64: &str = include_str!("../../shaders/lattice/su3_math_f64.wgsl");

/// Base SU(3) lattice matrix operations (includes storage bindings).
///
/// For composition-safe math-only version, use [`WGSL_SU3_MATH_F64`].
pub const WGSL_SU3_LATTICE_F64: &str = include_str!("../../shaders/lattice/su3_f64.wgsl");

/// PCG hash PRNG library: `pcg_hash → uniform_f64`.
///
/// Shared PRNG for all lattice kernels that need random number generation.
/// Stateless (hash-based), suitable for GPU-parallel use.
pub const WGSL_PRNG_PCG_F64: &str = include_str!("../../shaders/lattice/prng_pcg_f64.wgsl");

/// DF64 SU(3) gauge force (9.9× throughput via f32-pair arithmetic).
///
/// Uses neighbor-buffer pattern `nbr_buf[site*8+dir]` for indexing-agnostic
/// staple computation.
pub const WGSL_SU3_GAUGE_FORCE_DF64: &str =
    include_str!("../../shaders/lattice/su3_gauge_force_df64.wgsl");

/// DF64 kinetic energy: `T_link` via f32-pair arithmetic.
pub const WGSL_SU3_KINETIC_ENERGY_DF64: &str =
    include_str!("../../shaders/lattice/su3_kinetic_energy_df64.wgsl");

/// File-based BLAS axpy: y[i] += α × x[i] (f64).
///
/// Standalone version of the inline [`super::cg::WGSL_AXPY_F64`] for
/// multi-shader composition via file concatenation.
pub const WGSL_AXPY_FILE_F64: &str = include_str!("../../shaders/lattice/axpy_f64.wgsl");

/// File-based complex dot product Re(<a|b>) (f64).
///
/// Standalone version of the inline [`super::cg::WGSL_COMPLEX_DOT_RE_F64`].
pub const WGSL_COMPLEX_DOT_RE_FILE_F64: &str =
    include_str!("../../shaders/lattice/complex_dot_re_f64.wgsl");

/// File-based BLAS xpay: p[i] = x[i] + β × p[i] (f64).
///
/// Standalone version of the inline [`super::cg::WGSL_XPAY_F64`].
pub const WGSL_XPAY_FILE_F64: &str = include_str!("../../shaders/lattice/xpay_f64.wgsl");

// ── Multi-shift CG absorption (hotSpring, Mar 2026) ─────────────────

/// Multi-shift CG: Jegerlehner `ζ` recurrence + shifted `α`/`β` scalars.
///
/// Per-shift kernel that updates `ζ_prev`, `ζ_curr`, `α_s`, and `β_ratio`
/// from the base CG scalars (`α_j`, `β_{j-1}`, `α_{j-1}`).
pub const WGSL_MS_ZETA_UPDATE_F64: &str =
    include_str!("../../shaders/lattice/ms_zeta_update_f64.wgsl");

/// Multi-shift CG: `x_σ += α_σ × p_σ` (shifted solution update).
pub const WGSL_MS_X_UPDATE_F64: &str = include_str!("../../shaders/lattice/ms_x_update_f64.wgsl");

/// Multi-shift CG: `p_σ = ζ_σ × r + β_σ × p_σ` (shifted direction update).
///
/// `β_σ = ratio² × β_base` where `ratio = ζ_new / ζ_curr`.
pub const WGSL_MS_P_UPDATE_F64: &str = include_str!("../../shaders/lattice/ms_p_update_f64.wgsl");

/// Shifted CG scalar: `α = rz / (pAp + σ × pp)`.
///
/// Single-thread kernel for shifted CG systems `(D†D + σ)x = b`.
pub const WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64: &str =
    include_str!("../../shaders/lattice/cg_compute_alpha_shifted_f64.wgsl");

/// Shifted CG vector update: `x += α × p`, `r -= α × (Ap + σ × p)`.
///
/// Fuses the shift correction into the residual update.
pub const WGSL_CG_UPDATE_XR_SHIFTED_F64: &str =
    include_str!("../../shaders/lattice/cg_update_xr_shifted_f64.wgsl");

// ── GPU-resident Hamiltonian / Metropolis absorption (hotSpring, Mar 2026) ──

/// GPU-resident Hamiltonian assembly: `H = β×(6V − plaq_sum) + T + S_f`.
///
/// Single-thread kernel that eliminates CPU readback for Hamiltonian computation.
/// Outputs both `H` and per-sector diagnostics (`S_gauge`, `T`, `S_ferm`).
pub const WGSL_HAMILTONIAN_ASSEMBLY_F64: &str =
    include_str!("../../shaders/lattice/hamiltonian_assembly_f64.wgsl");

/// GPU-resident fermion action sum for one RHMC sector.
///
/// `S_f = α₀·dots[0] + Σ_s(α[s]·dots[s+1])`. Accumulates into `s_ferm` buffer
/// so multiple sectors can call sequentially.
pub const WGSL_FERMION_ACTION_SUM_F64: &str =
    include_str!("../../shaders/lattice/fermion_action_sum_f64.wgsl");

/// GPU-resident Metropolis accept/reject test with 9-entry diagnostics.
///
/// Computes `ΔH = H_new − H_old`, applies `min(1, exp(−ΔH))` criterion,
/// and writes acceptance flag + per-sector diagnostics in a single readback.
pub const WGSL_GPU_METROPOLIS_F64: &str =
    include_str!("../../shaders/lattice/gpu_metropolis_f64.wgsl");

// ── Sprint 22g: cross-spring deep absorption (hotSpring V0632, Mar 2026) ──

/// TMU-accelerated Box-Muller SU(3) momentum generation.
///
/// Offloads log/cos/sin transcendentals to TMU texture lookups; ALU handles
/// PCG hash, sqrt, and Gell-Mann construction only. Requires `prng_pcg_f64.wgsl`
/// prepended.
pub const WGSL_SU3_RANDOM_MOMENTA_TMU_F64: &str =
    include_str!("../../shaders/lattice/su3_random_momenta_tmu_f64.wgsl");

/// ROP-accelerated fermion force accumulation via `atomicAdd(i32)`.
///
/// Fixed-point scale 2^20 allows multiple poles to dispatch simultaneously
/// with no inter-pole barriers. A conversion kernel adds accumulated i32
/// values back to the f64 momentum buffer.
pub const WGSL_SU3_FERMION_FORCE_ACCUMULATE_ROP_F64: &str =
    include_str!("../../shaders/lattice/su3_fermion_force_accumulate_rop_f64.wgsl");

/// Fixed-point i32 → f64 conversion for ROP force accumulation.
///
/// Dispatched once after all fermion force poles have accumulated.
/// `mom[i] += f64(force_accum[i]) / scale_factor`.
pub const WGSL_SU3_FORCE_ATOMIC_TO_MOMENTUM_F64: &str =
    include_str!("../../shaders/lattice/su3_force_atomic_to_momentum_f64.wgsl");

/// Subgroup-accelerated f64 tree reduction.
///
/// Uses `subgroupAdd` for warp/wavefront-level reduction, then shared memory
/// for cross-subgroup accumulation. Requires `wgpu::Features::SUBGROUP`.
pub const WGSL_SUM_REDUCE_SUBGROUP_F64: &str =
    include_str!("../../shaders/reduce/sum_reduce_subgroup_f64.wgsl");

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

    fn assert_has_entry_point(name: &str, src: &str) {
        assert!(
            src.contains("fn main("),
            "{name} must contain fn main() entry point"
        );
    }

    fn assert_no_enable_f64(name: &str, src: &str) {
        assert!(
            !src.contains("enable f64;"),
            "{name} must not contain `enable f64;` — stripped by compile_shader_f64()"
        );
    }

    fn assert_agpl_header(name: &str, src: &str) {
        assert!(
            src.starts_with("// SPDX-License-Identifier: AGPL-3.0"),
            "{name} must start with AGPL SPDX header"
        );
    }

    #[test]
    fn s60_shaders_valid() {
        for (name, src) in [
            ("gauge_force", WGSL_SU3_GAUGE_FORCE_F64),
            ("kinetic_energy", WGSL_SU3_KINETIC_ENERGY_F64),
            ("link_update", WGSL_SU3_LINK_UPDATE_F64),
            ("momentum_update", WGSL_SU3_MOMENTUM_UPDATE_F64),
            ("random_momenta", WGSL_SU3_RANDOM_MOMENTA_F64),
            ("gaussian_fermion", WGSL_GAUSSIAN_FERMION_F64),
            ("staggered_fermion_force", WGSL_STAGGERED_FERMION_FORCE_F64),
        ] {
            assert_non_empty(name, src);
            assert_has_entry_point(name, src);
            assert_agpl_header(name, src);
        }
    }

    #[test]
    fn s64_shaders_valid() {
        for (name, src) in [
            ("gauge_force_df64", WGSL_SU3_GAUGE_FORCE_DF64),
            ("kinetic_energy_df64", WGSL_SU3_KINETIC_ENERGY_DF64),
        ] {
            assert_non_empty(name, src);
            assert_has_entry_point(name, src);
            assert_agpl_header(name, src);
        }
    }

    #[test]
    fn preamble_shaders_valid() {
        for (name, src) in [
            ("su3_math", WGSL_SU3_MATH_F64),
            ("su3_lattice", WGSL_SU3_LATTICE_F64),
            ("prng_pcg", WGSL_PRNG_PCG_F64),
        ] {
            assert_non_empty(name, src);
            assert_agpl_header(name, src);
        }
    }

    #[test]
    fn sprint_22g_shaders_valid() {
        for (name, src) in [
            ("tmu_momenta", WGSL_SU3_RANDOM_MOMENTA_TMU_F64),
            ("rop_force", WGSL_SU3_FERMION_FORCE_ACCUMULATE_ROP_F64),
            ("atomic_to_momentum", WGSL_SU3_FORCE_ATOMIC_TO_MOMENTUM_F64),
            ("subgroup_reduce", WGSL_SUM_REDUCE_SUBGROUP_F64),
        ] {
            assert_non_empty(name, src);
            assert_has_entry_point(name, src);
            assert_no_enable_f64(name, src);
            assert_agpl_header(name, src);
        }
    }

    #[test]
    fn tmu_momenta_requires_pcg_preamble() {
        assert!(
            WGSL_SU3_RANDOM_MOMENTA_TMU_F64.contains("uniform_f64"),
            "TMU momenta shader must use uniform_f64 from PCG preamble"
        );
    }

    #[test]
    fn rop_force_uses_atomic_add() {
        assert!(
            WGSL_SU3_FERMION_FORCE_ACCUMULATE_ROP_F64.contains("atomicAdd"),
            "ROP force shader must use atomicAdd"
        );
    }

    #[test]
    fn subgroup_reduce_uses_subgroup_add() {
        assert!(
            WGSL_SUM_REDUCE_SUBGROUP_F64.contains("subgroupAdd"),
            "subgroup reduce shader must use subgroupAdd"
        );
        let has_directive = WGSL_SUM_REDUCE_SUBGROUP_F64.lines().any(|l| {
            let trimmed = l.trim();
            !trimmed.starts_with("//") && trimmed.contains("enable subgroups;")
        });
        assert!(
            !has_directive,
            "must not contain enable subgroups; directive"
        );
    }
}
