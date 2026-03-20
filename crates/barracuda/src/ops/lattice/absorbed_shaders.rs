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
