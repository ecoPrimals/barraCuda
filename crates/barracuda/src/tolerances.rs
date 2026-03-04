// SPDX-License-Identifier: AGPL-3.0-or-later

//! Centralized validation tolerances with mathematical justification.
//!
//! Every tolerance threshold used in validation is defined here.
//! No ad-hoc magic numbers. Imitates the neuralSpring hotSpring pattern.

/// Tolerance descriptor with absolute and relative bounds plus justification.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub name: &'static str,
    pub abs_tol: f64,
    pub rel_tol: f64,
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

/// Shannon diversity index H' = -Σ p_i ln(p_i).
pub const BIO_DIVERSITY_SHANNON: Tolerance = Tolerance {
    name: "bio_diversity_shannon",
    abs_tol: 1e-8,
    rel_tol: 1e-8,
    justification: "log accumulation over S species; well-conditioned for p_i > 0",
};

/// Simpson diversity index 1 - Σ p_i².
pub const BIO_DIVERSITY_SIMPSON: Tolerance = Tolerance {
    name: "bio_diversity_simpson",
    abs_tol: 1e-10,
    rel_tol: 1e-10,
    justification: "sum-of-squares; exact integer arithmetic when from counts",
};

/// Phylogenetic distance metrics (UniFrac, patristic).
pub const BIO_PHYLOGENETIC: Tolerance = Tolerance {
    name: "bio_phylogenetic",
    abs_tol: 1e-6,
    rel_tol: 1e-4,
    justification: "branch-length accumulation over tree; float rounding per edge",
};

#[expect(clippy::unwrap_used, reason = "tests")]
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
    fn tolerances_have_finite_values() {
        let tols = [
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
        ];
        for t in &tols {
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
