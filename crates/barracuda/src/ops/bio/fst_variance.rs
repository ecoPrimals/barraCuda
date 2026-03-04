// SPDX-License-Identifier: AGPL-3.0-or-later

//! FST (Fixation Index) variance decomposition — Weir-Cockerham estimator.
//!
//! Computes Wright's F-statistics (FST, FIS, FIT) from allele frequencies
//! and population sample sizes using the Weir-Cockerham (1984) method.
//!
//! # Input
//!
//! - `allele_freqs`: allele frequency of the focal allele in each population (0..1)
//! - `population_sizes`: number of diploid individuals sampled per population
//!
//! # Output
//!
//! - `fst` (θ): proportion of variance among populations
//! - `f_is` (f): within-population inbreeding coefficient
//! - `f_it` (F): total inbreeding coefficient
//!
//! # References
//!
//! - Weir & Cockerham (1984) Evolution 38:1358-1370
//! - scikit-allel weir_cockerham_fst

use crate::error::{BarracudaError, Result};

/// Result of FST variance decomposition.
#[derive(Debug, Clone, Copy)]
pub struct FstResult {
    /// FST (θ): variance among populations / total variance
    pub fst: f64,
    /// FIS (f): within-population inbreeding coefficient
    pub f_is: f64,
    /// FIT (F): total inbreeding coefficient
    pub f_it: f64,
}

/// Compute FST variance decomposition using Weir-Cockerham estimator.
///
/// # Arguments
///
/// * `allele_freqs` - Allele frequency (0..1) of the focal allele in each population
/// * `population_sizes` - Number of diploid individuals sampled per population
///
/// # Returns
///
/// `FstResult` with `fst`, `f_is`, `f_it`. All three can be negative when
/// estimated from small samples (Weir-Cockerham corrects for sample size).
///
/// # Errors
///
/// Returns `InvalidInput` if lengths differ, sizes are zero, or frequencies invalid.
pub fn fst_variance_decomposition(
    allele_freqs: &[f64],
    population_sizes: &[usize],
) -> Result<FstResult> {
    let r = allele_freqs.len();
    if r != population_sizes.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "fst_variance_decomposition: allele_freqs len {} != population_sizes len {}",
                allele_freqs.len(),
                population_sizes.len()
            ),
        });
    }
    if r < 2 {
        return Err(BarracudaError::InvalidInput {
            message: "fst_variance_decomposition requires at least 2 populations".to_string(),
        });
    }

    let n: Vec<f64> = population_sizes.iter().map(|&s| s as f64).collect();
    let n_total: f64 = n.iter().sum();
    if n_total < 1.0 {
        return Err(BarracudaError::InvalidInput {
            message: "fst_variance_decomposition: total sample size must be >= 1".to_string(),
        });
    }

    for (i, &p) in allele_freqs.iter().enumerate() {
        if !(0.0..=1.0).contains(&p) {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "fst_variance_decomposition: allele_freqs[{i}] = {p} must be in [0,1]"
                ),
            });
        }
    }

    let n_bar = n.iter().sum::<f64>() / r as f64;
    if n_bar <= 1.0 {
        return Err(BarracudaError::InvalidInput {
            message: "fst_variance_decomposition: mean sample size must be > 1".to_string(),
        });
    }
    let n_c = (n_total - n.iter().map(|ni| ni * ni).sum::<f64>() / n_total) / (r - 1) as f64;
    if n_c <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: "fst_variance_decomposition: n_c <= 0 (check population sizes)".to_string(),
        });
    }

    // Weighted mean allele frequency
    let p_bar: f64 = allele_freqs
        .iter()
        .zip(n.iter())
        .map(|(p, ni)| p * ni)
        .sum::<f64>()
        / n_total;

    // Sample variance of allele frequencies over populations
    let s_squared: f64 = allele_freqs
        .iter()
        .zip(n.iter())
        .map(|(p, ni)| ni * (p - p_bar) * (p - p_bar))
        .sum::<f64>()
        / (n_bar * (r - 1) as f64);

    // Without genotype data we use expected heterozygosity under HWE
    let h_bar = 2.0 * p_bar * (1.0 - p_bar);

    // Weir-Cockerham variance components (biallelic, expected H under HWE)
    let a = (n_bar / n_c)
        * (s_squared
            - (1.0 / (n_bar - 1.0))
                * (p_bar * (1.0 - p_bar) - (r - 1) as f64 * s_squared / r as f64 - h_bar / 4.0));
    let b = (n_bar / (n_bar - 1.0))
        * (p_bar * (1.0 - p_bar)
            - (r - 1) as f64 * s_squared / r as f64
            - ((2.0 * n_bar - 1.0) * h_bar / (4.0 * n_bar)));
    let c = h_bar / 2.0;

    let denom = a + b + c;
    let fst = if denom.abs() > 1e-300 { a / denom } else { 0.0 };

    let denom_bc = b + c;
    let f_is = if denom_bc.abs() > 1e-300 {
        1.0 - c / denom_bc
    } else {
        0.0
    };

    let f_it = if denom.abs() > 1e-300 {
        1.0 - c / denom
    } else {
        0.0
    };

    Ok(FstResult { fst, f_is, f_it })
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_fixation_between_pops() {
        // Pop 1: p=0, Pop 2: p=1 → maximum differentiation
        let freqs = [0.0, 1.0];
        let sizes = [10, 10];
        let r = fst_variance_decomposition(&freqs, &sizes).expect("ok");
        assert!(
            r.fst > 0.9,
            "complete fixation should give FST ≈ 1, got {}",
            r.fst
        );
    }

    #[test]
    fn no_differentiation() {
        // All pops same frequency — Weir-Cockerham can yield small negative FST
        // with finite sample sizes; use large n to minimize the correction.
        let freqs = [0.5, 0.5, 0.5];
        let sizes = [1000, 1000, 1000];
        let r = fst_variance_decomposition(&freqs, &sizes).expect("ok");
        assert!(
            r.fst.abs() < 0.01,
            "no differentiation should give FST ≈ 0, got {}",
            r.fst
        );
    }

    #[test]
    fn moderate_differentiation() {
        // Different frequencies
        let freqs = [0.2, 0.5, 0.8];
        let sizes = [100, 100, 100];
        let r = fst_variance_decomposition(&freqs, &sizes).expect("ok");
        assert!(
            r.fst > 0.0 && r.fst < 1.0,
            "moderate diff: FST in (0,1), got {}",
            r.fst
        );
    }

    #[test]
    fn relationship_fis_fit_fst() {
        // (1-FIT) = (1-FIS)(1-FST)
        let freqs = [0.3, 0.7];
        let sizes = [50, 50];
        let r = fst_variance_decomposition(&freqs, &sizes).expect("ok");
        let lhs = 1.0 - r.f_it;
        let rhs = (1.0 - r.f_is) * (1.0 - r.fst);
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "(1-FIT) = (1-FIS)(1-FST): lhs={lhs}, rhs={rhs}"
        );
    }

    #[test]
    fn len_mismatch_rejected() {
        let r = fst_variance_decomposition(&[0.5, 0.5], &[10, 10, 10]);
        assert!(r.is_err());
    }

    #[test]
    fn single_pop_rejected() {
        let r = fst_variance_decomposition(&[0.5], &[10]);
        assert!(r.is_err());
    }

    #[test]
    fn invalid_freq_rejected() {
        let r = fst_variance_decomposition(&[0.5, 1.5], &[10, 10]);
        assert!(r.is_err());
    }
}
