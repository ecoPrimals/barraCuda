// SPDX-License-Identifier: AGPL-3.0-or-later

//! Runtime tolerance registry for cross-spring introspection.
//!
//! Springs can enumerate all known tolerances at runtime for validation
//! harnesses, documentation generation, and tolerance-tier auditing.

use super::precision::{
    PRECISION_BF16, PRECISION_BINARY, PRECISION_DF64, PRECISION_DF128, PRECISION_F16,
    PRECISION_F32, PRECISION_F64, PRECISION_F64_PRECISE, PRECISION_FP8_E4M3, PRECISION_FP8_E5M2,
    PRECISION_INT2, PRECISION_Q4, PRECISION_Q8, PRECISION_QF128, PRECISION_TF32,
};
use super::{
    ACCUMULATION, BIO_ALLELE_FREQ, BIO_DIVERSITY_SHANNON, BIO_DIVERSITY_SIMPSON, BIO_HMM,
    BIO_NUCLEOTIDE_DIVERSITY, BIO_PHYLOGENETIC, DETERMINISM, EQUILIBRIUM, HYDRO_CROP_COEFFICIENT,
    HYDRO_ET0, HYDRO_SOIL_MOISTURE, HYDRO_WATER_BALANCE, ITERATIVE, LATTICE_CG_FORCE,
    LATTICE_CG_METROPOLIS, LATTICE_FERMION_FORCE, LATTICE_METROPOLIS_DELTA_H, LATTICE_PLAQUETTE,
    LATTICE_RHMC_APPROX_ERROR, LINALG_FROBENIUS, LINALG_MATMUL, LINALG_TRANSPOSE, MACHINE,
    PHARMA_FOCE, PHARMA_HILL, PHARMA_MICHAELIS_MENTEN, PHARMA_NCA, PHARMA_POP_PK, PHARMA_SCFA,
    PHARMA_VPC, PHYSICS_ANDERSON_EIGENVALUE, PHYSICS_LATTICE_ACTION, PHYSICS_LYAPUNOV,
    REDUCTION_LOGSUMEXP, REDUCTION_MEAN, REDUCTION_SUM, REDUCTION_VARIANCE, SIGNAL_FFT, SIGNAL_QRS,
    SPECIAL_BESSEL, SPECIAL_ERF, SPECIAL_GAMMA, STATISTICAL, STOCHASTIC, TRANSCENDENTAL, Tolerance,
};

/// All registered tolerances, accessible for runtime introspection.
///
/// Springs use this to validate that their domain tolerances are consistent
/// with the tiered architecture (DETERMINISM <= MACHINE <= ... <= EQUILIBRIUM)
/// and to generate tolerance documentation automatically.
#[must_use]
pub fn all_tolerances() -> &'static [Tolerance] {
    &[
        LINALG_MATMUL,
        LINALG_TRANSPOSE,
        LINALG_FROBENIUS,
        REDUCTION_SUM,
        REDUCTION_MEAN,
        REDUCTION_VARIANCE,
        REDUCTION_LOGSUMEXP,
        BIO_HMM,
        BIO_ALLELE_FREQ,
        BIO_NUCLEOTIDE_DIVERSITY,
        SPECIAL_ERF,
        SPECIAL_GAMMA,
        SPECIAL_BESSEL,
        HYDRO_ET0,
        HYDRO_SOIL_MOISTURE,
        HYDRO_WATER_BALANCE,
        HYDRO_CROP_COEFFICIENT,
        PHYSICS_ANDERSON_EIGENVALUE,
        PHYSICS_LATTICE_ACTION,
        PHYSICS_LYAPUNOV,
        LATTICE_CG_FORCE,
        LATTICE_CG_METROPOLIS,
        LATTICE_RHMC_APPROX_ERROR,
        LATTICE_PLAQUETTE,
        LATTICE_FERMION_FORCE,
        LATTICE_METROPOLIS_DELTA_H,
        BIO_DIVERSITY_SHANNON,
        BIO_DIVERSITY_SIMPSON,
        BIO_PHYLOGENETIC,
        DETERMINISM,
        MACHINE,
        ACCUMULATION,
        TRANSCENDENTAL,
        ITERATIVE,
        STATISTICAL,
        STOCHASTIC,
        EQUILIBRIUM,
        PHARMA_FOCE,
        PHARMA_VPC,
        PHARMA_NCA,
        PHARMA_POP_PK,
        PHARMA_HILL,
        PHARMA_MICHAELIS_MENTEN,
        PHARMA_SCFA,
        SIGNAL_FFT,
        SIGNAL_QRS,
        PRECISION_DF128,
        PRECISION_QF128,
        PRECISION_F64_PRECISE,
        PRECISION_F64,
        PRECISION_DF64,
        PRECISION_F32,
        PRECISION_TF32,
        PRECISION_F16,
        PRECISION_BF16,
        PRECISION_FP8_E4M3,
        PRECISION_FP8_E5M2,
        PRECISION_Q8,
        PRECISION_Q4,
        PRECISION_INT2,
        PRECISION_BINARY,
    ]
}

/// Look up a tolerance by name at runtime.
///
/// Springs can query `tolerances::by_name("pharma_foce")` to get the
/// tolerance descriptor without importing the constant directly.
#[must_use]
pub fn by_name(name: &str) -> Option<&'static Tolerance> {
    all_tolerances().iter().find(|t| t.name == name)
}

/// Return the appropriate tiered tolerance for a given category.
///
/// Maps semantic categories to the tiered architecture:
/// - `"determinism"` -> [`DETERMINISM`]
/// - `"machine"` -> [`MACHINE`]
/// - `"accumulation"` -> [`ACCUMULATION`]
/// - `"transcendental"` -> [`TRANSCENDENTAL`]
/// - `"iterative"` -> [`ITERATIVE`]
/// - `"statistical"` -> [`STATISTICAL`]
/// - `"stochastic"` -> [`STOCHASTIC`]
/// - `"equilibrium"` -> [`EQUILIBRIUM`]
/// - anything else -> `None`
#[must_use]
pub fn tier(category: &str) -> Option<&'static Tolerance> {
    match category {
        "determinism" => Some(&DETERMINISM),
        "machine" => Some(&MACHINE),
        "accumulation" => Some(&ACCUMULATION),
        "transcendental" => Some(&TRANSCENDENTAL),
        "iterative" => Some(&ITERATIVE),
        "statistical" => Some(&STATISTICAL),
        "stochastic" => Some(&STOCHASTIC),
        "equilibrium" => Some(&EQUILIBRIUM),
        _ => None,
    }
}
