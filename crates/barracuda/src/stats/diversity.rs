// SPDX-License-Identifier: AGPL-3.0-only

//! Alpha and beta diversity metrics for ecological community analysis.
//!
//! CPU implementations of standard ecology metrics (QIIME2/skbio compatible).
//! For GPU-accelerated Shannon, Simpson, and Pielou, use
//! [`crate::ops::bio::diversity_fusion::DiversityFusionGpu`].
//!
//! Absorbed from wetSpring `bio/diversity.rs` (Feb 2026, Session 64).
//!
//! # Alpha diversity
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`observed_features`] | Count of non-zero species |
//! | [`shannon`] | Shannon entropy H' (natural log) |
//! | [`simpson`] | Simpson diversity index (1 - Σpᵢ²) |
//! | [`chao1`] | Chao1 richness estimator |
//! | [`pielou_evenness`] | Pielou's J' = H'/ln(S) |
//! | [`alpha_diversity`] | All metrics at once |
//!
//! # Beta diversity
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`bray_curtis`] | Bray-Curtis dissimilarity |
//! | [`bray_curtis_condensed`] | Condensed distance matrix |
//! | [`bray_curtis_matrix`] | Full symmetric distance matrix |
//!
//! # Rarefaction
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`rarefaction_curve`] | Expected species vs subsampling depth |

// ── Alpha diversity ─────────────────────────────────────────────────────

/// Observed features: count of non-zero entries.
#[inline]
#[must_use]
pub fn observed_features(counts: &[f64]) -> f64 {
    counts.iter().filter(|&&c| c > 0.0).count() as f64
}

/// Shannon entropy: H' = −Σ pᵢ ln(pᵢ) (natural log, matching skbio).
#[inline]
#[must_use]
pub fn shannon(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            h -= p * p.ln();
        }
    }
    h
}

/// Shannon entropy from pre-computed frequency proportions.
///
/// H' = −Σ pᵢ ln(pᵢ) where pᵢ are already normalized to sum to 1.
///
/// Unlike [`shannon`] which accepts raw counts and normalizes internally,
/// this function trusts the caller to provide valid frequencies (≥0, sum ≈ 1).
#[inline]
#[must_use]
pub fn shannon_from_frequencies(frequencies: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in frequencies {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Simpson diversity index: 1 − Σ pᵢ². Higher = more diverse (0 to 1).
#[inline]
#[must_use]
pub fn simpson(counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let sum_p2: f64 = counts
        .iter()
        .filter(|&&c| c > 0.0)
        .map(|&c| (c / total).powi(2))
        .sum();
    1.0 - sum_p2
}

/// Chao1 richness estimator.
///
/// Chao1 = `S_obs` + f1(f1−1) / (2(f2+1))
/// where f1 = singletons, f2 = doubletons.
///
/// Singleton/doubleton detection uses a half-width of 0.5 (integer counts).
#[inline]
#[must_use]
pub fn chao1(counts: &[f64]) -> f64 {
    const HALFWIDTH: f64 = 0.5;
    let s_obs = observed_features(counts);
    let f1 = counts
        .iter()
        .filter(|&&c| (c - 1.0).abs() < HALFWIDTH)
        .count() as f64;
    let f2 = counts
        .iter()
        .filter(|&&c| (c - 2.0).abs() < HALFWIDTH)
        .count() as f64;

    if f2 > 0.0 {
        s_obs + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))
    } else if f1 > 0.0 {
        s_obs + (f1 * (f1 - 1.0)) / 2.0
    } else {
        s_obs
    }
}

/// Chao 1984 estimator from integer counts (classic formula).
///
/// `S_chao1 = S_obs + f1²/(2·f2)` when f2 > 0,
/// else `S_obs + f1·(f1−1)/2` when f1 > 0.
///
/// This variant accepts `u64` counts directly, avoiding f64 allocation
/// for amplicon sequencing pipelines with integer abundance tables.
///
/// Provenance: groundSpring `rare_biosphere.rs` → toadStool absorption (S70).
#[inline]
#[must_use]
pub fn chao1_classic(counts: &[u64]) -> f64 {
    let s_obs = counts.iter().filter(|&&c| c > 0).count() as f64;
    let f1 = counts.iter().filter(|&&c| c == 1).count() as f64;
    let f2 = counts.iter().filter(|&&c| c == 2).count() as f64;

    if f2 > 0.0 {
        s_obs + (f1 * f1) / (2.0 * f2)
    } else if f1 > 0.0 {
        s_obs + f1 * (f1 - 1.0) / 2.0
    } else {
        s_obs
    }
}

/// Pielou's evenness: J' = H'/ln(S). Range [0, 1].
///
/// Returns 0.0 when S ≤ 1 (undefined).
#[inline]
#[must_use]
pub fn pielou_evenness(counts: &[f64]) -> f64 {
    let s = observed_features(counts);
    if s <= 1.0 {
        return 0.0;
    }
    shannon(counts) / s.ln()
}

/// Alpha diversity summary for a single sample.
#[derive(Debug, Clone)]
pub struct AlphaDiversity {
    /// Observed species count.
    pub observed: f64,
    /// Shannon entropy.
    pub shannon: f64,
    /// Simpson diversity index.
    pub simpson: f64,
    /// Chao1 richness estimator.
    pub chao1: f64,
    /// Pielou evenness.
    pub evenness: f64,
}

/// Compute all alpha diversity metrics for a sample.
#[must_use]
pub fn alpha_diversity(counts: &[f64]) -> AlphaDiversity {
    AlphaDiversity {
        observed: observed_features(counts),
        shannon: shannon(counts),
        simpson: simpson(counts),
        chao1: chao1(counts),
        evenness: pielou_evenness(counts),
    }
}

// ── Beta diversity ──────────────────────────────────────────────────────

/// Bray-Curtis dissimilarity: Σ|aᵢ−bᵢ| / Σ(aᵢ+bᵢ). Range [0, 1].
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
#[must_use]
pub fn bray_curtis(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let mut num = 0.0;
    let mut den = 0.0;
    for (&ai, &bi) in a.iter().zip(b) {
        num += (ai - bi).abs();
        den += ai + bi;
    }
    if den == 0.0 { 0.0 } else { num / den }
}

/// Condensed Bray-Curtis distance matrix (lower triangle, row-major).
///
/// For N samples, returns N*(N−1)/2 values.
/// Use [`condensed_index`] to look up a pair.
#[must_use]
pub fn bray_curtis_condensed(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..n {
        for j in 0..i {
            condensed.push(bray_curtis(&samples[i], &samples[j]));
        }
    }
    condensed
}

/// Index into a condensed distance matrix for pair (i, j).
///
/// # Panics
///
/// Panics if `i == j`.
#[inline]
#[must_use]
pub fn condensed_index(i: usize, j: usize) -> usize {
    assert_ne!(i, j, "diagonal entries are always zero");
    let (a, b) = if i > j { (i, j) } else { (j, i) };
    a * (a - 1) / 2 + b
}

/// Full symmetric Bray-Curtis distance matrix (N × N, row-major).
///
/// For large N, prefer [`bray_curtis_condensed`].
#[must_use]
pub fn bray_curtis_matrix(samples: &[Vec<f64>]) -> Vec<f64> {
    let n = samples.len();
    let mut matrix = vec![0.0; n * n];
    let condensed = bray_curtis_condensed(samples);
    for i in 1..n {
        for j in 0..i {
            let idx = condensed_index(i, j);
            matrix[i * n + j] = condensed[idx];
            matrix[j * n + i] = condensed[idx];
        }
    }
    matrix
}

// ── Rarefaction ─────────────────────────────────────────────────────────

/// Rarefaction curve: expected species at each subsampling depth.
///
/// Uses the exact hypergeometric formula (no randomness):
/// E[`S_n`] = S − Σ C(N−Nᵢ, n) / C(N, n)
#[must_use]
pub fn rarefaction_curve(counts: &[f64], depths: &[f64]) -> Vec<f64> {
    let total: f64 = counts.iter().sum();
    if total <= 0.0 {
        return vec![0.0; depths.len()];
    }
    let total_n = total as u64;

    depths
        .iter()
        .map(|&depth| {
            let n = depth.min(total) as u64;
            if n == 0 {
                return 0.0;
            }
            if n >= total_n {
                return observed_features(counts);
            }
            let mut expected = 0.0;
            for &c in counts {
                if c <= 0.0 {
                    continue;
                }
                let ni = c as u64;
                let absent_log = log_hypergeometric_absent(total_n, ni, n);
                expected += 1.0 - absent_log.exp();
            }
            expected
        })
        .collect()
}

/// log(C(N−Nᵢ, n) / C(N, n)) in log-space for numerical stability.
fn log_hypergeometric_absent(big_n: u64, ni: u64, n: u64) -> f64 {
    if ni >= big_n {
        return f64::NEG_INFINITY;
    }
    let remainder = big_n - ni;
    if n > remainder {
        return f64::NEG_INFINITY;
    }
    let mut log_ratio = 0.0_f64;
    for k in 0..n {
        log_ratio += ((remainder - k) as f64).ln() - ((big_n - k) as f64).ln();
    }
    log_ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shannon_uniform() {
        let counts = vec![25.0; 4];
        assert!((shannon(&counts) - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn shannon_single_species() {
        let counts = vec![100.0, 0.0, 0.0, 0.0];
        assert!(shannon(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn shannon_from_freq_uniform() {
        let freq = vec![0.25; 4];
        assert!((shannon_from_frequencies(&freq) - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn shannon_from_freq_matches_counts() {
        let counts = vec![10.0, 20.0, 30.0, 40.0];
        let total: f64 = counts.iter().sum();
        let freq: Vec<f64> = counts.iter().map(|c| c / total).collect();
        assert!((shannon(&counts) - shannon_from_frequencies(&freq)).abs() < 1e-10);
    }

    #[test]
    fn shannon_from_freq_single_species() {
        let freq = vec![1.0, 0.0, 0.0];
        assert!(shannon_from_frequencies(&freq).abs() < f64::EPSILON);
    }

    #[test]
    fn simpson_uniform() {
        let counts = vec![25.0; 4];
        assert!((simpson(&counts) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn simpson_single() {
        let counts = vec![100.0, 0.0, 0.0];
        assert!(simpson(&counts).abs() < f64::EPSILON);
    }

    #[test]
    fn observed() {
        let counts = vec![10.0, 0.0, 5.0, 0.0, 1.0];
        assert!((observed_features(&counts) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn chao1_with_singletons() {
        let counts = vec![10.0, 20.0, 30.0, 1.0, 1.0, 2.0];
        assert!((chao1(&counts) - 6.5).abs() < 1e-10);
    }

    #[test]
    fn chao1_no_singletons() {
        let counts = vec![10.0, 20.0, 30.0];
        assert!((chao1(&counts) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bray_curtis_identical() {
        let a = vec![10.0, 20.0, 30.0];
        assert!(bray_curtis(&a, &a).abs() < f64::EPSILON);
    }

    #[test]
    fn bray_curtis_disjoint() {
        let a = vec![10.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 10.0];
        assert!((bray_curtis(&a, &b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn bray_curtis_symmetry() {
        let a = vec![10.0, 20.0, 30.0, 0.0, 5.0];
        let b = vec![15.0, 10.0, 25.0, 5.0, 0.0];
        assert!((bray_curtis(&a, &b) - bray_curtis(&b, &a)).abs() < 1e-15);
    }

    #[test]
    fn pielou_uniform() {
        let counts = vec![25.0; 4];
        assert!((pielou_evenness(&counts) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rarefaction_at_full_depth() {
        let counts = vec![10.0, 20.0, 30.0, 5.0];
        let total: f64 = counts.iter().sum();
        let curve = rarefaction_curve(&counts, &[total]);
        assert!((curve[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn rarefaction_monotonic() {
        let counts = vec![50.0, 30.0, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0];
        let depths: Vec<f64> = (1..=120).map(f64::from).collect();
        let curve = rarefaction_curve(&counts, &depths);
        for i in 1..curve.len() {
            assert!(
                curve[i] >= curve[i - 1] - 1e-10,
                "not monotonic at depth {}",
                depths[i]
            );
        }
    }

    #[test]
    fn alpha_diversity_uniform() {
        let counts = vec![25.0; 4];
        let ad = alpha_diversity(&counts);
        assert!((ad.observed - 4.0).abs() < f64::EPSILON);
        assert!((ad.shannon - 4.0_f64.ln()).abs() < 1e-10);
        assert!((ad.simpson - 0.75).abs() < 1e-10);
        assert!((ad.evenness - 1.0).abs() < 1e-10);
    }

    #[test]
    fn condensed_index_symmetric() {
        assert_eq!(condensed_index(2, 0), condensed_index(0, 2));
        assert_eq!(condensed_index(3, 1), condensed_index(1, 3));
    }

    #[test]
    fn condensed_vs_full_matrix() {
        let samples = vec![vec![10.0, 20.0], vec![15.0, 25.0]];
        let dm = bray_curtis_matrix(&samples);
        assert!(dm[0].abs() < f64::EPSILON);
        assert!(dm[3].abs() < f64::EPSILON);
        assert!((dm[1] - dm[2]).abs() < f64::EPSILON);
    }
}
