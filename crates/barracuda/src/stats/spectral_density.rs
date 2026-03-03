// SPDX-License-Identifier: AGPL-3.0-only

//! Spectral density estimation and Random Matrix Theory bounds.
//!
//! - [`empirical_spectral_density`]: histogram of eigenvalues (normalized)
//! - [`marchenko_pastur_bounds`]: RMT bounds for singular-value spectrum
//!
//! Provenance: neuralSpring baseCamp `weight_spectral.rs` (Feb 2026)

/// Empirical spectral density: histogram of eigenvalues into `n_bins` bins.
///
/// Returns `(bin_centers, bin_counts)` normalized so that counts sum to 1.
///
/// # Panics
///
/// Does not panic; returns empty vectors for empty input or zero bins.
#[must_use]
pub fn empirical_spectral_density(eigenvalues: &[f64], n_bins: usize) -> (Vec<f64>, Vec<f64>) {
    if eigenvalues.is_empty() || n_bins == 0 {
        return (vec![], vec![]);
    }

    let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let bin_width = if range < 1e-300 {
        1.0
    } else {
        range / n_bins as f64
    };

    let mut counts = vec![0.0_f64; n_bins];
    let n_total = eigenvalues.len() as f64;
    for &ev in eigenvalues {
        let idx = ((ev - min) / bin_width) as usize;
        let idx = idx.min(n_bins - 1);
        counts[idx] += 1.0 / n_total;
    }

    let centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5).mul_add(bin_width, min))
        .collect();

    (centers, counts)
}

/// Marchenko-Pastur bounds for the expected spectral density of random matrices.
///
/// For a random matrix with aspect ratio γ = rows/cols, the eigenvalue
/// distribution of M^T M / n converges to the Marchenko-Pastur law with
/// support [λ_min, λ_max].
///
/// Returns `(λ_min, λ_max)` where:
/// - `λ_min = (1 - √γ)²`
/// - `λ_max = (1 + √γ)²`
#[must_use]
pub fn marchenko_pastur_bounds(gamma: f64) -> (f64, f64) {
    let sq = gamma.sqrt();
    let lambda_min = (1.0 - sq).powi(2);
    let lambda_max = (1.0 + sq).powi(2);
    (lambda_min, lambda_max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esd_sums_to_one() {
        let eigenvalues: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let (centers, counts) = empirical_spectral_density(&eigenvalues, 10);
        assert_eq!(centers.len(), 10);
        assert_eq!(counts.len(), 10);
        let total: f64 = counts.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "ESD should sum to 1, got {total}"
        );
    }

    #[test]
    fn esd_empty_input() {
        let (c, d) = empirical_spectral_density(&[], 10);
        assert!(c.is_empty());
        assert!(d.is_empty());
    }

    #[test]
    fn esd_zero_bins() {
        let (c, d) = empirical_spectral_density(&[1.0, 2.0], 0);
        assert!(c.is_empty());
        assert!(d.is_empty());
    }

    #[test]
    fn esd_single_value() {
        let (_centers, counts) = empirical_spectral_density(&[5.0; 20], 4);
        assert_eq!(counts.len(), 4);
        let total: f64 = counts.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mp_bounds_unit_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(1.0);
        assert!((lo - 0.0).abs() < 1e-10, "λ_min(γ=1) = 0, got {lo}");
        assert!((hi - 4.0).abs() < 1e-10, "λ_max(γ=1) = 4, got {hi}");
    }

    #[test]
    fn mp_bounds_quarter_aspect() {
        let (lo, hi) = marchenko_pastur_bounds(0.25);
        let expected_lo = (1.0 - 0.5_f64).powi(2);
        let expected_hi = (1.0 + 0.5_f64).powi(2);
        assert!(
            (lo - expected_lo).abs() < 1e-10,
            "λ_min(γ=0.25) = {expected_lo}, got {lo}"
        );
        assert!(
            (hi - expected_hi).abs() < 1e-10,
            "λ_max(γ=0.25) = {expected_hi}, got {hi}"
        );
    }

    #[test]
    fn mp_bounds_large_gamma() {
        let (lo, hi) = marchenko_pastur_bounds(4.0);
        assert!(lo > 0.0);
        assert!(hi > lo);
    }
}
