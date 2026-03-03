// SPDX-License-Identifier: AGPL-3.0-or-later

//! Level spacing statistics and band detection.
//!
//! Mean level spacing ratio ⟨r⟩ distinguishes Poisson (localized) from GOE
//! (extended) statistics. Band detection groups eigenvalues by gaps.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)
//! Spectral phase classification: neuralSpring V69 handoff

use crate::stats::marchenko_pastur_bounds;

/// Spectral phase based on outlier fraction beyond Marchenko-Pastur upper bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpectralPhase {
    /// < 5% outliers beyond MP upper bound
    Bulk,
    /// 5–20% outliers
    EdgeOfChaos,
    /// > 20% outliers
    Chaotic,
}

/// Spectral bandwidth = max(eigenvalues) - min(eigenvalues). Returns 0.0 if empty.
#[must_use]
pub fn spectral_bandwidth(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return 0.0;
    }
    let (min, max) = eigenvalues
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });
    max - min
}

/// Condition number = max(|eigenvalues|) / min(|eigenvalues|). Returns f64::INFINITY if min is zero.
#[must_use]
pub fn spectral_condition_number(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.is_empty() {
        return f64::INFINITY;
    }
    let (min_abs, max_abs) = eigenvalues
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &x| {
            let a = x.abs();
            (lo.min(a), hi.max(a))
        });
    if min_abs < 1e-300 {
        f64::INFINITY
    } else {
        max_abs / min_abs
    }
}

/// Classify spectral phase by outlier fraction beyond Marchenko-Pastur upper bound.
#[must_use]
pub fn classify_spectral_phase(eigenvalues: &[f64], marchenko_upper: f64) -> SpectralPhase {
    if eigenvalues.is_empty() {
        return SpectralPhase::Bulk;
    }
    let outliers = eigenvalues.iter().filter(|&&x| x > marchenko_upper).count();
    let frac = outliers as f64 / eigenvalues.len() as f64;
    if frac < 0.05 {
        SpectralPhase::Bulk
    } else if frac <= 0.20 {
        SpectralPhase::EdgeOfChaos
    } else {
        SpectralPhase::Chaotic
    }
}

/// Spectral analysis result with bandwidth, condition number, and phase.
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    pub eigenvalues: Vec<f64>,
    pub bandwidth: f64,
    pub condition_number: f64,
    pub phase: SpectralPhase,
    pub marchenko_upper: f64,
}

impl SpectralAnalysis {
    /// Build from eigenvalues and aspect ratio γ = n/p for Marchenko-Pastur bounds.
    #[must_use]
    pub fn from_eigenvalues(eigenvalues: Vec<f64>, gamma: f64) -> Self {
        let (_lo, marchenko_upper) = marchenko_pastur_bounds(gamma);
        let bandwidth = spectral_bandwidth(&eigenvalues);
        let condition_number = spectral_condition_number(&eigenvalues);
        let phase = classify_spectral_phase(&eigenvalues, marchenko_upper);
        Self {
            eigenvalues,
            bandwidth,
            condition_number,
            phase,
            marchenko_upper,
        }
    }
}

/// Compute the mean level spacing ratio ⟨r⟩ from sorted eigenvalues.
///
/// r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
/// where s_i = λ_{i+1} − λ_i.
///
/// Known values:
/// - Poisson (localized): ⟨r⟩ = 2 ln 2 − 1 ≈ 0.3863
/// - GOE (extended + time-reversal): ⟨r⟩ ≈ 0.5307
///
/// # Provenance
/// Oganesyan & Huse (2007), Phys. Rev. B 75, 155111
/// Atas et al. (2013), Phys. Rev. Lett. 110, 084101
pub fn level_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    let n = eigenvalues.len();
    if n < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n - 2 {
        let s1 = eigenvalues[i + 1] - eigenvalues[i];
        let s2 = eigenvalues[i + 2] - eigenvalues[i + 1];
        if s1 > 0.0 && s2 > 0.0 {
            let r = s1.min(s2) / s1.max(s2);
            sum += r;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Poisson level spacing ratio (localized states).
pub const POISSON_R: f64 = 0.386_294_361_119_890_6; // 2 ln 2 - 1

/// GOE level spacing ratio (extended states with time-reversal symmetry).
pub const GOE_R: f64 = 0.5307;

/// Detect spectral bands from sorted eigenvalues.
///
/// Groups eigenvalues into bands separated by gaps. A "gap" is defined as a
/// spacing exceeding `gap_factor` times the median spacing. Returns a vector
/// of (band_min, band_max) pairs.
pub fn detect_bands(eigenvalues: &[f64], gap_factor: f64) -> Vec<(f64, f64)> {
    if eigenvalues.len() < 2 {
        if eigenvalues.len() == 1 {
            return vec![(eigenvalues[0], eigenvalues[0])];
        }
        return Vec::new();
    }

    let mut spacings: Vec<f64> = eigenvalues.windows(2).map(|w| w[1] - w[0]).collect();
    spacings.sort_by(f64::total_cmp);
    let median = spacings[spacings.len() / 2];

    let threshold = median * gap_factor;
    let mut bands = Vec::new();
    let mut band_start = eigenvalues[0];

    for w in eigenvalues.windows(2) {
        if w[1] - w[0] > threshold {
            bands.push((band_start, w[0]));
            band_start = w[1];
        }
    }
    if let Some(&last) = eigenvalues.last() {
        bands.push((band_start, last));
    }

    bands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{anderson_hamiltonian, find_all_eigenvalues};

    #[test]
    fn level_spacing_poisson() {
        let (d, e) = anderson_hamiltonian(1000, 8.0, 42);
        let evals = find_all_eigenvalues(&d, &e);
        let r = level_spacing_ratio(&evals);
        assert!(
            (r - POISSON_R).abs() < 0.05,
            "Strong disorder: r={r:.4}, expected Poisson={POISSON_R:.4}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn level_spacing_ratio_two_eigenvalues() {
        let evals = vec![0.0, 1.0];
        let r = level_spacing_ratio(&evals);
        assert_eq!(r, 0.0, "n<3 should return 0");
    }

    #[test]
    fn level_spacing_ratio_equally_spaced() {
        let evals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let r = level_spacing_ratio(&evals);
        assert!((r - 1.0).abs() < 1e-10, "equal spacing gives r=1");
    }

    #[test]
    fn detect_bands_no_gap() {
        let evals: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let bands = detect_bands(&evals, 10.0);
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0], (0.0, 19.0));
    }

    #[test]
    fn detect_bands_large_gap() {
        let mut evals: Vec<f64> = (0..5).map(|i| i as f64).collect();
        evals.extend((100..105).map(|i| i as f64));
        let bands = detect_bands(&evals, 2.0);
        assert!(bands.len() >= 2);
    }

    #[test]
    fn spectral_bandwidth_basic() {
        assert_eq!(spectral_bandwidth(&[1.0, 2.0, 5.0]), 4.0);
        assert_eq!(spectral_bandwidth(&[]), 0.0);
    }

    #[test]
    fn spectral_condition_number_basic() {
        assert!((spectral_condition_number(&[0.5, 1.0, 2.0]) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn classify_spectral_phase_bulk() {
        // All eigenvalues below MP upper (4.0 for γ=1) → Bulk
        let evals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        assert_eq!(classify_spectral_phase(&evals, 4.0), SpectralPhase::Bulk);
    }

    #[test]
    fn classify_spectral_phase_chaotic() {
        // >20% above 4.0 → Chaotic (e.g. 5 of 20 = 25%)
        let mut evals: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
        evals.extend([5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(classify_spectral_phase(&evals, 4.0), SpectralPhase::Chaotic);
    }

    #[test]
    fn spectral_analysis_from_eigenvalues() {
        let evals = vec![1.0, 2.0, 5.0];
        let a = SpectralAnalysis::from_eigenvalues(evals.clone(), 1.0);
        assert_eq!(a.bandwidth, 4.0);
        assert!((a.condition_number - 5.0).abs() < 1e-10); // max=5, min=1
        assert_eq!(a.eigenvalues, evals);
    }
}
