// SPDX-License-Identifier: AGPL-3.0-or-later

//! SpectralNautilusBridge — maps spectral analysis features to NautilusBrain observations.
//!
//! Bridges `barracuda::spectral` eigenvalue statistics to `NautilusBrain`'s
//! `BetaObservation` input format, enabling spectral-driven evolutionary
//! reservoir predictions.
//!
//! Provenance: neuralSpring S102 `nautilus_bridge.rs` → toadStool absorption

use super::brain::BetaObservation;

/// Spectral features extracted from eigenvalue data.
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Mean level spacing ratio ⟨r⟩ (GOE ≈ 0.531, Poisson ≈ 0.386).
    pub level_spacing_ratio: f64,
    /// Spectral bandwidth (max − min eigenvalue).
    pub bandwidth: f64,
    /// Condition number (max / min absolute eigenvalue, if min > 0).
    pub condition_number: Option<f64>,
    /// Minimum eigenvalue (localization probe).
    pub lambda_min: f64,
    /// Phase classification from ⟨r⟩.
    pub phase: SpectralPhase,
}

/// Phase classification based on level spacing ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralPhase {
    /// Extended (metallic): ⟨r⟩ > 0.50
    Bulk,
    /// Critical: 0.42 < ⟨r⟩ < 0.50
    EdgeOfChaos,
    /// Localized (insulating): ⟨r⟩ < 0.42
    Localized,
}

impl SpectralFeatures {
    /// Extract spectral features from a sorted eigenvalue array.
    pub fn from_eigenvalues(eigenvalues: &[f64]) -> Self {
        let n = eigenvalues.len();
        if n < 2 {
            return Self {
                level_spacing_ratio: 0.0,
                bandwidth: 0.0,
                condition_number: None,
                lambda_min: eigenvalues.first().copied().unwrap_or(0.0),
                phase: SpectralPhase::Localized,
            };
        }

        let bandwidth = eigenvalues[n - 1] - eigenvalues[0];
        let lambda_min = eigenvalues[0];

        let abs_min = eigenvalues
            .iter()
            .map(|v| v.abs())
            .fold(f64::INFINITY, f64::min);
        let abs_max = eigenvalues.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let condition_number = if abs_min > 1e-15 {
            Some(abs_max / abs_min)
        } else {
            None
        };

        let r = level_spacing_ratio_from_sorted(eigenvalues);

        let phase = if r > 0.50 {
            SpectralPhase::Bulk
        } else if r > 0.42 {
            SpectralPhase::EdgeOfChaos
        } else {
            SpectralPhase::Localized
        };

        Self {
            level_spacing_ratio: r,
            bandwidth,
            condition_number,
            lambda_min,
            phase,
        }
    }

    /// Convert spectral features to a `BetaObservation` for NautilusBrain.
    ///
    /// Maps spectral quantities to physics-analogous fields:
    /// - `beta` ← level_spacing_ratio (control parameter analog)
    /// - `plaquette` ← bandwidth (order parameter analog)
    /// - `cg_iters` ← condition_number (computational cost proxy)
    /// - `acceptance` ← 1.0 (always valid)
    /// - `delta_h_abs` ← 0.0 (no energy change for spectral obs)
    /// - `anderson_r` ← level_spacing_ratio (direct)
    /// - `anderson_lambda_min` ← lambda_min
    pub fn to_observation(&self) -> BetaObservation {
        BetaObservation {
            beta: self.level_spacing_ratio,
            plaquette: self.bandwidth,
            cg_iters: self.condition_number.unwrap_or(1.0),
            acceptance: 1.0,
            delta_h_abs: 0.0,
            quenched_plaq: None,
            quenched_plaq_var: None,
            anderson_r: Some(self.level_spacing_ratio),
            anderson_lambda_min: Some(self.lambda_min),
        }
    }
}

fn level_spacing_ratio_from_sorted(eigs: &[f64]) -> f64 {
    let n = eigs.len();
    if n < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;
    for i in 1..(n - 1) {
        let s_prev = eigs[i] - eigs[i - 1];
        let s_next = eigs[i + 1] - eigs[i];
        if s_prev > 0.0 && s_next > 0.0 {
            sum += s_prev.min(s_next) / s_prev.max(s_next);
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_features_from_sorted() {
        let eigs = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let feat = SpectralFeatures::from_eigenvalues(&eigs);
        assert!((feat.bandwidth - 4.5).abs() < 1e-12);
        assert!((feat.lambda_min - 0.5).abs() < 1e-12);
        assert!(feat.level_spacing_ratio > 0.9, "uniform spacing → r ≈ 1.0");
    }

    #[test]
    fn spectral_features_to_observation() {
        let eigs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let feat = SpectralFeatures::from_eigenvalues(&eigs);
        let obs = feat.to_observation();
        assert!((obs.beta - feat.level_spacing_ratio).abs() < 1e-15);
        assert!(obs.anderson_r.is_some());
        assert!(obs.anderson_lambda_min.is_some());
    }

    #[test]
    fn spectral_phase_classification() {
        let extended_eigs: Vec<f64> = (0..100)
            .map(|i| i as f64 + 0.5 * (i as f64).sin())
            .collect();
        let feat = SpectralFeatures::from_eigenvalues(&extended_eigs);
        assert_ne!(feat.phase, SpectralPhase::Localized);
    }
}
