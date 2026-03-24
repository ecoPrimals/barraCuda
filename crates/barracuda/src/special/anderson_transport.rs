// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anderson localization transport and conductance primitives.
//!
//! Landauer formula: G = (e²/h) × T, where T is the transmission coefficient.
//! For 1D Anderson model: log(T) ∝ -L/ξ(E,W), where ξ is the localization length.
//!
//! Provenance: wetSpring V20

use crate::error::{BarracudaError, Result};

/// Thouless-formula approximation for the localization length ξ(E,W).
///
/// For the 1D Anderson model with disorder strength W and energy E:
/// ξ(E,W) ≈ 105 × (4 - E²).max(0.01) / (W² + 0.01)
///
/// - At band center (E≈0): ξ is largest; states are least localized.
/// - At band edges (|E|→2): ξ decreases; stronger localization.
/// - With disorder W: ξ decreases as W² increases.
///
/// # Arguments
///
/// * `disorder_strength` - W, the half-width of the uniform disorder distribution
/// * `energy` - E, the single-particle energy (band center E=0 for 1D tight-binding)
///
/// # Returns
///
/// Localization length in units of lattice spacing.
#[must_use]
pub fn localization_length(disorder_strength: f64, energy: f64) -> f64 {
    let w_sq = disorder_strength.mul_add(disorder_strength, 0.01);
    let band_factor = energy.mul_add(-energy, 4.0).max(0.01);
    105.0 * band_factor / w_sq
}

/// Anderson conductance (normalized to conductance quantum G₀ = e²/h).
///
/// Uses the Landauer formula: G = (e²/h) × T, with transmission
/// T = exp(-L/ξ) for a 1D chain of length L.
///
/// Returns the normalized conductance G/G₀ = T.
///
/// # Arguments
///
/// * `disorder_strength` - W, half-width of disorder
/// * `system_size` - L, chain length in lattice sites
/// * `energy` - E, single-particle energy
///
/// # Returns
///
/// Normalized conductance in [0, 1]. Ballistic (W=0) → ~1.0; localized → 0.
///
/// # Errors
///
/// Returns [`Err`] if `disorder_strength` is negative.
pub fn anderson_conductance(
    disorder_strength: f64,
    system_size: usize,
    energy: f64,
) -> Result<f64> {
    if disorder_strength < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("disorder_strength must be non-negative, got {disorder_strength}"),
        });
    }

    let xi = localization_length(disorder_strength, energy);
    let l = system_size as f64;

    if xi <= 0.0 {
        return Ok(0.0);
    }

    let transmission = (-l / xi).exp();
    Ok(transmission.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_disorder_conductance_ballistic() {
        let g = anderson_conductance(0.0, 100, 0.0).unwrap();
        assert!(
            g > 0.99,
            "Zero disorder should give ballistic conductance ~1.0, got {g}"
        );
    }

    #[test]
    fn strong_disorder_conductance_localized() {
        let g = anderson_conductance(20.0, 1000, 0.0).unwrap();
        assert!(
            g < 0.01,
            "Strong disorder should give conductance → 0, got {g}"
        );
    }

    #[test]
    fn localization_length_decreases_with_disorder() {
        let xi_small = localization_length(1.0, 0.0);
        let xi_large = localization_length(10.0, 0.0);
        assert!(
            xi_large < xi_small,
            "Localization length should decrease with disorder: ξ(W=10)={xi_large} < ξ(W=1)={xi_small}"
        );
    }

    #[test]
    fn localization_length_positive() {
        assert!(localization_length(0.0, 0.0) > 0.0);
        assert!(localization_length(5.0, 1.0) > 0.0);
    }

    #[test]
    fn anderson_conductance_invalid_disorder() {
        let r = anderson_conductance(-1.0, 100, 0.0);
        assert!(r.is_err());
    }
}
