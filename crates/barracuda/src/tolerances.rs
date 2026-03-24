// SPDX-License-Identifier: AGPL-3.0-or-later

//! Centralized validation tolerances with mathematical justification.
//!
//! Every tolerance threshold used in validation is defined here.
//! No ad-hoc magic numbers. Imitates the neuralSpring hotSpring pattern.
//!
//! # Stability Contract
//!
//! All tolerance constants in this module are **SemVer-stable**: tightening
//! a tolerance (lowering `abs_tol` or `rel_tol`) or removing a constant is a
//! **breaking change** requiring a major version bump or `BREAKING_CHANGES.md`
//! entry. Loosening a tolerance (raising values) is a minor/patch change.
//!
//! Springs depend on these constants for reproducible validation harnesses.
//! If a tolerance needs to change, file a `wateringHole` handoff first so
//! all consuming springs can update their expectations.

/// Tolerance descriptor with absolute and relative bounds plus justification.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Identifier for this tolerance.
    pub name: &'static str,
    /// Absolute tolerance threshold.
    pub abs_tol: f64,
    /// Relative tolerance threshold.
    pub rel_tol: f64,
    /// Mathematical justification for the chosen values.
    pub justification: &'static str,
}

/// Check whether `computed` matches `expected` within the tolerance.
///
/// Uses combined absolute-or-relative: passes if
/// `|computed - expected| <= abs_tol` or
/// `|computed - expected| <= rel_tol * max(|expected|, 1.0)`.
#[must_use]
pub fn check(computed: f64, expected: f64, tol: &Tolerance) -> bool {
    if !computed.is_finite() || !expected.is_finite() {
        return computed == expected;
    }
    let abs = (computed - expected).abs();
    if abs <= tol.abs_tol {
        return true;
    }
    let scale = expected.abs().max(1.0);
    abs <= tol.rel_tol * scale
}

// ═══════════════════════════════════════════════════════════════════
// linalg tolerances
// ═══════════════════════════════════════════════════════════════════

/// Matmul: dot-product accumulation O(√n) rounding for inner dim n.
pub const LINALG_MATMUL: Tolerance = Tolerance {
    name: "linalg_matmul",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "f64 dot-product accumulation; √n rounding for inner dim n",
};

/// Transpose: pure data movement, no arithmetic.
pub const LINALG_TRANSPOSE: Tolerance = Tolerance {
    name: "linalg_transpose",
    abs_tol: 1e-14,
    rel_tol: 1e-14,
    justification: "exact data movement; only f64 representation",
};

/// Frobenius norm: single-pass reduction.
pub const LINALG_FROBENIUS: Tolerance = Tolerance {
    name: "linalg_frobenius",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "f64 sum-of-squares reduction; accumulation order",
};

// ═══════════════════════════════════════════════════════════════════
// reduction tolerances
// ═══════════════════════════════════════════════════════════════════

/// Sum: Kahan or simple accumulation.
pub const REDUCTION_SUM: Tolerance = Tolerance {
    name: "reduction_sum",
    abs_tol: 1e-12,
    rel_tol: 1e-12,
    justification: "f64 accumulation; O(n) rounding for n elements",
};

/// Mean: sum / n.
pub const REDUCTION_MEAN: Tolerance = Tolerance {
    name: "reduction_mean",
    abs_tol: 1e-12,
    rel_tol: 1e-12,
    justification: "f64 sum then division; machine precision",
};

/// Variance: two-pass mean then residual sum.
pub const REDUCTION_VARIANCE: Tolerance = Tolerance {
    name: "reduction_variance",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "two-pass mean; catastrophic cancellation in subtraction",
};

/// Logsumexp: max-subtract + exp + sum + log.
pub const REDUCTION_LOGSUMEXP: Tolerance = Tolerance {
    name: "reduction_logsumexp",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "numerically stable; exp/log round-trip",
};

// ═══════════════════════════════════════════════════════════════════
// bio tolerances
// ═══════════════════════════════════════════════════════════════════

/// HMM forward: log-likelihood from T matrix-vector products.
pub const BIO_HMM: Tolerance = Tolerance {
    name: "bio_hmm",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "forward-backward accumulates rounding from T steps",
};

/// Allele frequency: per-locus variance across populations.
pub const BIO_ALLELE_FREQ: Tolerance = Tolerance {
    name: "bio_allele_freq",
    abs_tol: 1e-6,
    rel_tol: 1e-6,
    justification: "mean/variance over populations; f64 two-pass",
};

/// Nucleotide diversity: pairwise differences.
pub const BIO_NUCLEOTIDE_DIVERSITY: Tolerance = Tolerance {
    name: "bio_nucleotide_diversity",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "pairwise counting; exact arithmetic on integers",
};

// ═══════════════════════════════════════════════════════════════════
// special tolerances
// ═══════════════════════════════════════════════════════════════════

/// erf: polynomial/Chebyshev approximation.
pub const SPECIAL_ERF: Tolerance = Tolerance {
    name: "special_erf",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "A&S 7.1.26; ~6 digits accuracy in f64",
};

/// Gamma: Lanczos approximation.
pub const SPECIAL_GAMMA: Tolerance = Tolerance {
    name: "special_gamma",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "Lanczos; ~12 digits for x in [0.5, 2]",
};

/// Bessel: polynomial approximations.
pub const SPECIAL_BESSEL: Tolerance = Tolerance {
    name: "special_bessel",
    abs_tol: 1e-6,
    rel_tol: 1e-6,
    justification: "A&S 9.4.1-9.4.6; ~6 digits in f64",
};

// ═══════════════════════════════════════════════════════════════════
// hydrology tolerances (airSpring, wetSpring cross-spring)
// ═══════════════════════════════════════════════════════════════════

/// ET₀ reference evapotranspiration (Penman-Monteith / Hargreaves).
pub const HYDRO_ET0: Tolerance = Tolerance {
    name: "hydro_et0",
    abs_tol: 0.05,
    rel_tol: 1e-3,
    justification: "FAO-56 PM: ~0.05 mm/day measurement uncertainty; GPU f64 chain",
};

/// Soil moisture (volumetric water content θ).
pub const HYDRO_SOIL_MOISTURE: Tolerance = Tolerance {
    name: "hydro_soil_moisture",
    abs_tol: 1e-4,
    rel_tol: 1e-3,
    justification: "Richards/water balance: iterative solver convergence + sensor noise",
};

/// Water balance (daily depletion/surplus accounting).
pub const HYDRO_WATER_BALANCE: Tolerance = Tolerance {
    name: "hydro_water_balance",
    abs_tol: 0.1,
    rel_tol: 1e-3,
    justification: "cascaded ET₀→Kc→WB accumulates rounding over seasonal pipeline",
};

/// Crop coefficient (Kc interpolation).
pub const HYDRO_CROP_COEFFICIENT: Tolerance = Tolerance {
    name: "hydro_crop_coefficient",
    abs_tol: 1e-6,
    rel_tol: 1e-6,
    justification: "linear interpolation between Kc_prev and Kc_next; exact on f64",
};

// ═══════════════════════════════════════════════════════════════════
// physics tolerances (hotSpring, groundSpring cross-spring)
// ═══════════════════════════════════════════════════════════════════

/// Anderson eigenvalue (Sturm bisection or Lanczos).
pub const PHYSICS_ANDERSON_EIGENVALUE: Tolerance = Tolerance {
    name: "physics_anderson_eigenvalue",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "Sturm bisection converges to machine precision for well-separated eigenvalues",
};

/// Lattice gauge action density.
pub const PHYSICS_LATTICE_ACTION: Tolerance = Tolerance {
    name: "physics_lattice_action",
    abs_tol: 1e-6,
    rel_tol: 1e-4,
    justification: "plaquette trace accumulation over lattice volume; GPU f64 reduction order",
};

/// Lyapunov exponent from transfer matrix.
pub const PHYSICS_LYAPUNOV: Tolerance = Tolerance {
    name: "physics_lyapunov",
    abs_tol: 1e-4,
    rel_tol: 1e-3,
    justification: "log(norm) accumulation over L steps; statistical averaging over disorder",
};

// ═══════════════════════════════════════════════════════════════════
// diversity tolerances (wetSpring cross-spring)
// ═══════════════════════════════════════════════════════════════════

/// Shannon diversity index H' = -Σ `p_i` `ln(p_i)`.
pub const BIO_DIVERSITY_SHANNON: Tolerance = Tolerance {
    name: "bio_diversity_shannon",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "log accumulation over S species; well-conditioned for p_i > 0",
};

/// Simpson diversity index 1 - Σ `p_i²`.
pub const BIO_DIVERSITY_SIMPSON: Tolerance = Tolerance {
    name: "bio_diversity_simpson",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "sum-of-squares; exact integer arithmetic when from counts",
};

/// Phylogenetic distance metrics (`UniFrac`, patristic).
pub const BIO_PHYLOGENETIC: Tolerance = Tolerance {
    name: "bio_phylogenetic",
    abs_tol: 1e-6,
    rel_tol: 1e-4,
    justification: "branch-length accumulation over tree; float rounding per edge",
};

// ═══════════════════════════════════════════════════════════════════
// Tiered tolerance architecture (absorbed from groundSpring V74)
//
// 13 tiers from strictest (machine precision) to loosest (equilibrium
// convergence).  Each tier has a semantic name and documented purpose.
// Springs select the tier that matches their physics, not an ad-hoc number.
// ═══════════════════════════════════════════════════════════════════

/// Deterministic operations: bit-exact or rounding error only.
pub const DETERMINISM: Tolerance = Tolerance {
    name: "tol::DETERMINISM",
    abs_tol: 0.0,
    rel_tol: f64::EPSILON,
    justification: "pure data movement or integer arithmetic; no floating-point accumulation",
};

/// Machine precision: single f64 operation.
pub const MACHINE: Tolerance = Tolerance {
    name: "tol::MACHINE",
    abs_tol: 1e-15,
    rel_tol: 1e-15,
    justification: "one f64 multiply/add; unit roundoff ~1.1e-16",
};

/// Accumulation: O(sqrt(n)) rounding from n f64 additions.
pub const ACCUMULATION: Tolerance = Tolerance {
    name: "tol::ACCUMULATION",
    abs_tol: 1e-12,
    rel_tol: 1e-12,
    justification: "sum/dot-product of ~10^6 terms; error ~sqrt(n)·eps",
};

/// Transcendental: exp, log, sin, cos minimax polynomials.
pub const TRANSCENDENTAL: Tolerance = Tolerance {
    name: "tol::TRANSCENDENTAL",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "range-reduced minimax with ~12 correct digits",
};

/// Iterative solver: CG, `BiCGSTAB` convergence criterion.
pub const ITERATIVE: Tolerance = Tolerance {
    name: "tol::ITERATIVE",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "iterative solver residual; problem-dependent conditioning",
};

/// Statistical: variance, correlation, bootstrap estimation.
pub const STATISTICAL: Tolerance = Tolerance {
    name: "tol::STATISTICAL",
    abs_tol: 1e-6,
    rel_tol: 1e-4,
    justification: "finite-sample statistics; dominated by sampling error not arithmetic",
};

/// Stochastic: PRNG-dependent quantities (Monte Carlo averages).
pub const STOCHASTIC: Tolerance = Tolerance {
    name: "tol::STOCHASTIC",
    abs_tol: 1e-3,
    rel_tol: 1e-2,
    justification: "Monte Carlo; error ~1/sqrt(N_samples); PRNG seed sensitive",
};

/// Equilibrium: convergence of SCF/MD thermalization.
pub const EQUILIBRIUM: Tolerance = Tolerance {
    name: "tol::EQUILIBRIUM",
    abs_tol: 1.0,
    rel_tol: 1e-2,
    justification: "thermodynamic equilibrium; fluctuations dominate",
};

// ═══════════════════════════════════════════════════════════════════
// Epsilon guards (absorbed from groundSpring V74)
//
// Named constants for division/log/sqrt guards that prevent
// catastrophic cancellation or division-by-zero.  Use these instead
// of ad-hoc "1e-15" or "1e-10" literals scattered through code.
// ═══════════════════════════════════════════════════════════════════

/// Epsilon guard constants for safe arithmetic.
pub mod eps {
    /// Minimum denominator for safe division.
    pub const SAFE_DIV: f64 = 1e-15;

    /// Floor for SSA (stochastic simulation) propensity sums.
    pub const SSA_FLOOR: f64 = 1e-30;

    /// Underflow guard for `exp()` arguments.
    pub const UNDERFLOW: f64 = -700.0;

    /// Overflow guard for `exp()` arguments.
    pub const OVERFLOW: f64 = 700.0;

    /// Guard for `log()` arguments (must be > 0).
    pub const LOG_FLOOR: f64 = 1e-300;

    /// Guard for `sqrt()` arguments (must be >= 0).
    pub const SQRT_FLOOR: f64 = 0.0;

    /// Guard for density values (must be non-negative).
    pub const DENSITY_FLOOR: f64 = 1e-20;

    /// Guard for probability values (must be in [0, 1]).
    pub const PROB_FLOOR: f64 = 1e-15;

    /// Overflow-safe midpoint: `(a + b) / 2` without overflow.
    ///
    /// Uses `a + (b - a) / 2` which is safe even when `a + b` overflows.
    #[must_use]
    pub fn midpoint(a: f64, b: f64) -> f64 {
        (b - a).mul_add(0.5, a)
    }
}

// ═══════════════════════════════════════════════════════════════════
// pharmacology tolerances (healthSpring cross-spring)
// ═══════════════════════════════════════════════════════════════════

/// FOCE population PK objective function.
pub const PHARMA_FOCE: Tolerance = Tolerance {
    name: "pharma_foce",
    abs_tol: 1e-6,
    rel_tol: 1e-4,
    justification: "FOCE likelihood: per-subject gradient accumulation; Newton step convergence",
};

/// VPC simulation envelope.
pub const PHARMA_VPC: Tolerance = Tolerance {
    name: "pharma_vpc",
    abs_tol: 1e-4,
    rel_tol: 1e-3,
    justification: "VPC Monte Carlo: 1000+ sims; dominated by sampling variance",
};

/// NCA AUC trapezoidal integration.
pub const PHARMA_NCA: Tolerance = Tolerance {
    name: "pharma_nca",
    abs_tol: 1e-8,
    rel_tol: 1e-6,
    justification: "trapezoidal AUC: exact for linear interpolation; f64 accumulation",
};

// ═══════════════════════════════════════════════════════════════════
// signal processing tolerances (healthSpring, neuralSpring)
// ═══════════════════════════════════════════════════════════════════

/// FFT spectral analysis.
pub const SIGNAL_FFT: Tolerance = Tolerance {
    name: "signal_fft",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "radix-2 FFT: O(n log n) butterfly; twiddle factor precision",
};

/// QRS detection (Pan-Tompkins).
pub const SIGNAL_QRS: Tolerance = Tolerance {
    name: "signal_qrs",
    abs_tol: 1e-3,
    rel_tol: 1e-2,
    justification: "streaming filter bank; adaptive threshold; dominated by signal noise",
};

// ═══════════════════════════════════════════════════════════════════
// Runtime tolerance registry (cross-spring introspection pattern)
//
// Springs can enumerate all known tolerances at runtime for validation
// harnesses, documentation generation, and tolerance-tier auditing.
// ═══════════════════════════════════════════════════════════════════

/// All registered tolerances, accessible for runtime introspection.
///
/// Springs use this to validate that their domain tolerances are consistent
/// with the tiered architecture (DETERMINISM ≤ MACHINE ≤ ... ≤ EQUILIBRIUM)
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
        SIGNAL_FFT,
        SIGNAL_QRS,
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
/// - `"determinism"` → [`DETERMINISM`]
/// - `"machine"` → [`MACHINE`]
/// - `"accumulation"` → [`ACCUMULATION`]
/// - `"transcendental"` → [`TRANSCENDENTAL`]
/// - `"iterative"` → [`ITERATIVE`]
/// - `"statistical"` → [`STATISTICAL`]
/// - `"stochastic"` → [`STOCHASTIC`]
/// - `"equilibrium"` → [`EQUILIBRIUM`]
/// - anything else → `None`
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_abs_tol() {
        assert!(check(1.0, 1.0 + 1e-15, &LINALG_TRANSPOSE));
        assert!(!check(1.0, 1.0 + 1e-10, &LINALG_TRANSPOSE));
    }

    #[test]
    fn check_rel_tol() {
        assert!(check(100.0, 100.0 + 1e-8, &LINALG_MATMUL));
        assert!(!check(100.0, 100.0 + 1e-5, &LINALG_MATMUL));
    }

    #[test]
    fn check_zero_expected() {
        assert!(check(1e-15, 0.0, &LINALG_TRANSPOSE));
    }

    #[test]
    fn check_nan_rejects() {
        assert!(!check(f64::NAN, 1.0, &LINALG_MATMUL));
        assert!(!check(1.0, f64::NAN, &LINALG_MATMUL));
    }

    #[test]
    fn check_infinity() {
        assert!(check(f64::INFINITY, f64::INFINITY, &LINALG_MATMUL));
    }

    #[test]
    fn tiered_tolerances_ordered() {
        const { assert!(DETERMINISM.abs_tol <= MACHINE.abs_tol) };
        const { assert!(MACHINE.abs_tol <= ACCUMULATION.abs_tol) };
        const { assert!(ACCUMULATION.abs_tol <= TRANSCENDENTAL.abs_tol) };
        const { assert!(TRANSCENDENTAL.abs_tol <= ITERATIVE.abs_tol) };
        const { assert!(ITERATIVE.abs_tol <= STATISTICAL.abs_tol) };
        const { assert!(STATISTICAL.abs_tol <= STOCHASTIC.abs_tol) };
        const { assert!(STOCHASTIC.abs_tol <= EQUILIBRIUM.abs_tol) };
    }

    #[test]
    fn eps_midpoint_safe() {
        assert_eq!(eps::midpoint(0.0, 10.0), 5.0);
        assert_eq!(eps::midpoint(-1.0, 1.0), 0.0);
        let large = f64::MAX * 0.5;
        let result = eps::midpoint(large, large);
        assert!(result.is_finite());
    }

    #[test]
    fn all_tolerances_registry() {
        let all = all_tolerances();
        assert!(all.len() >= 30, "registry should have 30+ tolerances");
        for t in all {
            assert!(!t.name.is_empty());
            assert!(!t.justification.is_empty());
            assert!(t.abs_tol.is_finite());
            assert!(t.rel_tol.is_finite());
        }
    }

    #[test]
    fn by_name_lookup() {
        assert_eq!(by_name("pharma_foce").unwrap().name, "pharma_foce");
        assert_eq!(by_name("signal_fft").unwrap().name, "signal_fft");
        assert!(by_name("nonexistent").is_none());
    }

    #[test]
    fn tier_lookup() {
        assert_eq!(tier("determinism").unwrap().name, "tol::DETERMINISM");
        assert_eq!(tier("equilibrium").unwrap().name, "tol::EQUILIBRIUM");
        assert!(tier("nonexistent").is_none());
    }

    #[test]
    fn eps_guards_positive() {
        const { assert!(eps::SAFE_DIV > 0.0) };
        const { assert!(eps::LOG_FLOOR > 0.0) };
        const { assert!(eps::PROB_FLOOR > 0.0) };
    }

    #[test]
    fn tolerances_have_finite_values() {
        for t in all_tolerances() {
            assert!(t.abs_tol.is_finite(), "{} abs_tol must be finite", t.name);
            assert!(t.rel_tol.is_finite(), "{} rel_tol must be finite", t.name);
            assert!(!t.name.is_empty(), "tolerance name must not be empty");
            assert!(
                !t.justification.is_empty(),
                "justification must not be empty"
            );
        }
    }
}
