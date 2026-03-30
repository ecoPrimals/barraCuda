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
    /// &lt; 5% outliers beyond MP upper bound (localized).
    Bulk,
    /// 5–20% outliers (edge of chaos).
    EdgeOfChaos,
    /// &gt; 20% outliers (chaotic).
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

/// Condition number = max(|eigenvalues|) / min(|eigenvalues|). Returns `f64::INFINITY` if min is zero.
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
    /// Eigenvalues (sorted).
    pub eigenvalues: Vec<f64>,
    /// Spectral bandwidth (max - min).
    pub bandwidth: f64,
    /// Condition number (max|λ| / min|λ|).
    pub condition_number: f64,
    /// Classified spectral phase.
    pub phase: SpectralPhase,
    /// Marchenko-Pastur upper bound used for phase classification.
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

/// Weight matrix spectral analysis — combines eigenvalue decomposition with
/// all spectral diagnostics in a single call.
///
/// Intended for neural network weight matrices where the full spectral
/// profile (bandwidth, condition number, phase, IPR, LSR) is needed.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct WeightMatrixAnalysis {
    /// Core spectral analysis (eigenvalues, bandwidth, condition number, phase).
    pub spectral: SpectralAnalysis,
    /// Mean inverse participation ratio: `Σ|ψ_i|⁴` averaged over eigenvectors.
    /// High IPR → localized; low IPR → extended.
    pub mean_ipr: f64,
    /// Mean level spacing ratio ⟨r⟩ (Poisson ≈ 0.386, GOE ≈ 0.531).
    pub level_spacing_ratio: f64,
    /// Spectral entropy: `−Σ p_i ln(p_i)` where `p_i = |λ_i| / Σ|λ_j|`.
    pub spectral_entropy: f64,
}

/// Analyze a symmetric weight matrix in one call.
///
/// Computes eigenvalues via CPU Jacobi eigensolve, then derives bandwidth,
/// condition number, phase classification, mean IPR, level spacing ratio,
/// and spectral entropy.
///
/// `gamma` is the aspect ratio n/p for Marchenko-Pastur phase bounds
/// (use 1.0 for square matrices).
///
/// # Errors
///
/// Returns [`Err`] if the eigenvalue decomposition fails.
#[cfg(feature = "gpu")]
pub fn analyze_weight_matrix(
    matrix: &[f64],
    n: usize,
    gamma: f64,
) -> crate::error::Result<WeightMatrixAnalysis> {
    let decomp = crate::linalg::eigh::eigh_f64(matrix, n)?;

    let mut sorted_evals = decomp.eigenvalues.clone();
    sorted_evals.sort_by(f64::total_cmp);

    let spectral = SpectralAnalysis::from_eigenvalues(sorted_evals.clone(), gamma);
    let lsr = level_spacing_ratio(&sorted_evals);
    let mean_ipr = compute_mean_ipr(&decomp.eigenvectors, n);
    let spectral_entropy = compute_spectral_entropy(&sorted_evals);

    Ok(WeightMatrixAnalysis {
        spectral,
        mean_ipr,
        level_spacing_ratio: lsr,
        spectral_entropy,
    })
}

#[cfg(feature = "gpu")]
fn compute_mean_ipr(eigenvectors: &[f64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut total_ipr = 0.0;
    for col in 0..n {
        let mut ipr = 0.0;
        for row in 0..n {
            let v = eigenvectors[row * n + col];
            ipr += v * v * v * v;
        }
        total_ipr += ipr;
    }
    total_ipr / n as f64
}

#[cfg_attr(not(feature = "gpu"), expect(dead_code))]
fn compute_spectral_entropy(eigenvalues: &[f64]) -> f64 {
    let total: f64 = eigenvalues.iter().map(|x| x.abs()).sum();
    if total < 1e-300 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &ev in eigenvalues {
        let p = ev.abs() / total;
        if p > 1e-300 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Compute the mean level spacing ratio ⟨r⟩ from sorted eigenvalues.
///
/// `r_i` = `min(s_i`, s_{i+1}) / `max(s_i`, s_{i+1})
/// where `s_i` = λ_{i+1} − `λ_i`.
///
/// Known values:
/// - Poisson (localized): ⟨r⟩ = 2 ln 2 − 1 ≈ 0.3863
/// - GOE (extended + time-reversal): ⟨r⟩ ≈ 0.5307
///
/// # Provenance
/// Oganesyan & Huse (2007), Phys. Rev. B 75, 155111
/// Atas et al. (2013), Phys. Rev. Lett. 110, 084101
#[must_use]
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

    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Poisson level spacing ratio (localized states).
pub const POISSON_R: f64 = 0.386_294_361_119_890_6; // 2 ln 2 - 1

/// GOE level spacing ratio (extended states with time-reversal symmetry).
pub const GOE_R: f64 = 0.5307;

/// Detect spectral bands from sorted eigenvalues.
///
/// Groups eigenvalues into bands separated by gaps. A "gap" is defined as a
/// spacing exceeding `gap_factor` times the median spacing. Returns a vector
/// of (`band_min`, `band_max`) pairs.
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
    #[expect(clippy::float_cmp, reason = "tests")]
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

    #[cfg(feature = "gpu")]
    #[test]
    fn analyze_weight_matrix_identity() {
        #[rustfmt::skip]
        let mat = [
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        let analysis = analyze_weight_matrix(&mat, 3, 1.0).unwrap();
        assert!((analysis.spectral.bandwidth - 2.0).abs() < 1e-6);
        assert!((analysis.spectral.condition_number - 3.0).abs() < 1e-6);
        assert!(analysis.mean_ipr > 0.0);
        assert!(analysis.spectral_entropy > 0.0);
        assert!(analysis.level_spacing_ratio >= 0.0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn analyze_weight_matrix_2x2() {
        #[rustfmt::skip]
        let mat = [
            2.0, 1.0,
            1.0, 2.0,
        ];
        let analysis = analyze_weight_matrix(&mat, 2, 1.0).unwrap();
        assert!((analysis.spectral.bandwidth - 2.0).abs() < 1e-6);
        assert_eq!(analysis.spectral.eigenvalues.len(), 2);
    }

    #[test]
    fn spectral_entropy_uniform() {
        let evals = vec![1.0, 1.0, 1.0, 1.0];
        let entropy = compute_spectral_entropy(&evals);
        let expected = (4.0_f64).ln();
        assert!(
            (entropy - expected).abs() < 1e-10,
            "uniform spectrum: entropy={entropy}, expected={expected}"
        );
    }

    #[test]
    fn spectral_entropy_single() {
        let evals = vec![5.0];
        let entropy = compute_spectral_entropy(&evals);
        assert!(entropy.abs() < 1e-10, "single eigenvalue → zero entropy");
    }
}
