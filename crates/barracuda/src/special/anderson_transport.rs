// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anderson localization transport, conductance, and selective permeability.
//!
//! ## Standard transport
//!
//! Landauer formula: G = (e²/h) × T, where T is the transmission coefficient.
//! For 1D Anderson model: log(T) ∝ -L/ξ(E,W), where ξ is the localization length.
//!
//! ## Selective permeability (Wave 167)
//!
//! Standard Anderson asks "does the signal propagate?" — binary.
//! Selective permeability asks "which signal types pass through this membrane
//! and which are blocked?" — each mode ω sees different effective disorder W_ω
//! and dimension d_ω through the same physical membrane.
//!
//! The permeability tensor P(ω, W_ω, d_ω) generalizes the scalar transmission
//! coefficient T to mode-dependent transport. Membrane selectivity
//! S = max(P) − min(P) measures the quality of biological filtering:
//! S = 0 for a rock (blocks everything) or breached barrier (passes everything);
//! S → 1 for a healthy cell membrane or well-tuned firewall.
//!
//! Provenance: wetSpring V20 (transport), Wave 167 (permeability)

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
        return Err(BarracudaError::invalid_input(format!(
            "disorder_strength must be non-negative, got {disorder_strength}"
        )));
    }

    let xi = localization_length(disorder_strength, energy);
    let l = system_size as f64;

    if xi <= 0.0 {
        return Ok(0.0);
    }

    let transmission = (-l / xi).exp();
    Ok(transmission.clamp(0.0, 1.0))
}

// ═══════════════════════════════════════════════════════════════════
// Selective Permeability — Wave 167
//
// Each signal mode ω sees different effective disorder W_ω and effective
// dimension d_ω through the same physical membrane. The transmission
// is mode-dependent: P(ω) = T(W_ω, d_ω, L, E).
// ═══════════════════════════════════════════════════════════════════

/// A single mode's view of a membrane — its effective Anderson parameters.
///
/// In a biological membrane, each signal type (nutrient, pathogen, cytokine,
/// etc.) encounters different effective disorder and dimensionality. A glucose
/// transporter creates low-W channels for glucose (extended, P→1) while
/// presenting high-W barriers to pathogens (localized, P→0).
#[derive(Debug, Clone, Copy)]
pub struct MembraneMode {
    /// Signal type identifier (application-defined).
    pub mode_id: u32,
    /// Effective disorder this mode sees through the membrane.
    pub w_eff: f64,
    /// Effective dimensionality this mode experiences (can be fractional).
    /// d_eff = 1.0 for 1D channels, 2.0 for 2D barriers, 3.0 for 3D bulk.
    pub d_eff: f64,
    /// System size (membrane thickness in lattice units).
    pub system_size: usize,
    /// Energy parameter for the mode.
    pub energy: f64,
}

/// The permeability of a single mode through a membrane.
///
/// Computed from the mode's effective Anderson parameters. The transmission
/// uses a dimension-dependent generalization of the Landauer formula:
///
/// - d ≤ 2: all states localize for W > 0 (Abrahams 1979), ξ is finite
/// - d ≥ 3: localization length diverges at W < W_c (metal-insulator transition)
///
/// For fractional d_eff, we interpolate the localization length between
/// integer dimensions to model partially breached barriers (e.g., AD skin
/// scratching promotes d_eff from 2.0 toward 3.0).
///
/// # Arguments
///
/// * `mode` - The mode's effective Anderson parameters
///
/// # Returns
///
/// Transmission coefficient P ∈ [0, 1]. P→1 means the mode passes freely
/// (extended state). P→0 means the mode is blocked (localized state).
///
/// # Errors
///
/// Returns [`Err`] if `w_eff` is negative.
pub fn membrane_permeability(mode: &MembraneMode) -> Result<f64> {
    if mode.w_eff < 0.0 {
        return Err(BarracudaError::invalid_input(format!(
            "w_eff must be non-negative, got {}",
            mode.w_eff
        )));
    }

    let xi = dimensional_localization_length(mode.w_eff, mode.energy, mode.d_eff);
    let l = mode.system_size as f64;

    if xi <= 0.0 {
        return Ok(0.0);
    }

    let transmission = (-l / xi).exp();
    Ok(transmission.clamp(0.0, 1.0))
}

/// Localization length generalized to fractional effective dimension.
///
/// Physics by dimension (Anderson 1958, Abrahams 1979, Slevin & Ohtsuki 1999):
///
/// - d=1: ξ(W) ≈ 105(4−E²)/W² — all states localize for any W > 0
/// - d=2: ξ(W) ~ l_mfp × exp(π k_F l_mfp) — all states localize but ξ can
///   be exponentially large (weak localization). For practical finite systems,
///   2D appears metallic at low W.
/// - d=3: genuine metal-insulator transition at W_c ≈ 16.5. Below W_c, band-
///   center states are truly EXTENDED (ξ → ∞). Above W_c, all states localize
///   with ξ ~ (W/W_c − 1)^(−ν), ν ≈ 1.57.
///
/// For fractional d_eff, log(ξ) is interpolated between adjacent integer
/// dimensions. This models partially breached barriers (AD skin dimensional
/// promotion, disturbed soil pore networks).
///
/// # Key invariant
///
/// At fixed disorder W > 0: ξ(d=1) < ξ(d=2) < ξ(d=3).
/// Higher dimension → more paths for signals → harder to localize.
#[must_use]
pub fn dimensional_localization_length(disorder: f64, energy: f64, d_eff: f64) -> f64 {
    let d_eff = d_eff.max(1.0);

    // 1D: all states localize. Thouless formula.
    let xi_1d = localization_length(disorder, energy);

    if d_eff <= 1.0 {
        return xi_1d;
    }

    // 2D: weak localization — ξ grows exponentially but remains finite.
    // ξ₂D ~ l_mfp × exp(π kF l_mfp). Approximate with kF l_mfp ~ ξ₁D/L₀.
    // Capped to prevent overflow while preserving the ordering ξ₂D >> ξ₁D.
    let xi_2d = if disorder > 1e-10 {
        let exponent = (core::f64::consts::PI * (xi_1d / 10.0).min(30.0)).min(30.0);
        xi_1d * exponent.exp()
    } else {
        1e15
    };

    if d_eff <= 2.0 {
        let frac = d_eff - 1.0;
        let log_xi = xi_1d.ln().mul_add(1.0 - frac, xi_2d.ln() * frac);
        return log_xi.exp();
    }

    // 3D: metal-insulator transition at W_c ≈ 16.5.
    // Below W_c: band-center states are truly EXTENDED — ξ diverges.
    // Above W_c: ξ ~ ξ₀ × (W/W_c − 1)^(−ν), ν ≈ 1.57 (Slevin & Ohtsuki 1999).
    const W_C: f64 = 16.5;
    const NU: f64 = 1.57;
    const XI_EXTENDED: f64 = 1e15;
    let xi_3d = if disorder < 1e-10 {
        XI_EXTENDED
    } else if disorder < W_C {
        // Below W_c: states are extended at band center.
        // ξ effectively infinite — signal propagates freely.
        XI_EXTENDED
    } else if (disorder - W_C).abs() < 0.01 {
        // At the critical point: ξ is large but finite for finite systems.
        xi_2d * 10.0
    } else {
        // Above W_c: localized regime. Critical scaling from insulating side.
        let reduced_w = disorder / W_C - 1.0;
        let xi_0 = xi_1d.max(1.0);
        xi_0 * reduced_w.powf(-NU).max(0.1)
    };

    if d_eff <= 3.0 {
        let frac = d_eff - 2.0;
        let log_2d = xi_2d.ln();
        let log_3d = xi_3d.ln();
        let log_xi = log_2d.mul_add(1.0 - frac, log_3d * frac);
        return log_xi.exp();
    }

    // d > 3: even more delocalized. Use 3D result as lower bound.
    xi_3d
}

/// Result of a multi-mode membrane permeability measurement.
#[derive(Debug, Clone)]
pub struct MembraneProfile {
    /// Per-mode transmission coefficients P(ω) ∈ [0, 1].
    pub transmissions: Vec<f64>,
    /// Selectivity S = max(P) − min(P). Measures membrane quality.
    /// S = 0: non-selective (rock or breached barrier).
    /// S → 1: maximally selective (healthy cell membrane).
    pub selectivity: f64,
    /// Mean transmission across all modes.
    pub mean_transmission: f64,
}

/// Compute the permeability profile of a membrane across multiple signal modes.
///
/// Each mode ω sees different effective disorder W_ω and dimension d_ω.
/// The membrane's selectivity S = max(P) − min(P) measures how well it
/// discriminates between signal types.
///
/// # Biological examples
///
/// - **Gut lining**: P(nutrient)→1, P(pathogen)→0 → S≈1 (selective)
/// - **Healthy epidermis**: P(water_vapor)→0.8, P(cytokine)→0.1 → S≈0.7
/// - **AD-scratched skin**: P(all)→0.9 → S≈0 (barrier breach, non-selective)
/// - **RustDesk relay**: P(commensal)→1, P(scanner)→0, P(operator_ns)→0 → S≈1
///
/// # Errors
///
/// Returns [`Err`] if any mode has negative `w_eff`.
pub fn membrane_profile(modes: &[MembraneMode]) -> Result<MembraneProfile> {
    if modes.is_empty() {
        return Ok(MembraneProfile {
            transmissions: vec![],
            selectivity: 0.0,
            mean_transmission: 0.0,
        });
    }

    let mut transmissions = Vec::with_capacity(modes.len());
    for mode in modes {
        transmissions.push(membrane_permeability(mode)?);
    }

    let p_min = transmissions.iter().cloned().fold(f64::INFINITY, f64::min);
    let p_max = transmissions
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let selectivity = p_max - p_min;
    let mean_transmission = transmissions.iter().sum::<f64>() / transmissions.len() as f64;

    Ok(MembraneProfile {
        transmissions,
        selectivity,
        mean_transmission,
    })
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

    // ── Selective permeability tests ──

    #[test]
    fn membrane_high_disorder_blocks() {
        let mode = MembraneMode {
            mode_id: 0,
            w_eff: 25.0,
            d_eff: 1.0,
            system_size: 500,
            energy: 0.0,
        };
        let p = membrane_permeability(&mode).unwrap();
        assert!(
            p < 0.01,
            "High disorder in 1D should block: P = {p}"
        );
    }

    #[test]
    fn membrane_low_disorder_passes() {
        let mode = MembraneMode {
            mode_id: 0,
            w_eff: 0.1,
            d_eff: 3.0,
            system_size: 10,
            energy: 0.0,
        };
        let p = membrane_permeability(&mode).unwrap();
        assert!(
            p > 0.5,
            "Low disorder in 3D should pass: P = {p}"
        );
    }

    #[test]
    fn membrane_2d_localizes_more_than_3d() {
        let mode_2d = MembraneMode {
            mode_id: 0,
            w_eff: 8.0,
            d_eff: 2.0,
            system_size: 50,
            energy: 0.0,
        };
        let mode_3d = MembraneMode {
            mode_id: 1,
            w_eff: 8.0,
            d_eff: 3.0,
            system_size: 50,
            energy: 0.0,
        };
        let p_2d = membrane_permeability(&mode_2d).unwrap();
        let p_3d = membrane_permeability(&mode_3d).unwrap();
        assert!(
            p_3d > p_2d,
            "3D should be more permeable than 2D at same W: P_3d={p_3d}, P_2d={p_2d}"
        );
    }

    #[test]
    fn membrane_selectivity_gut_lining() {
        let modes = vec![
            MembraneMode {
                mode_id: 0, // nutrient: low W, 3D
                w_eff: 0.5,
                d_eff: 3.0,
                system_size: 10,
                energy: 0.0,
            },
            MembraneMode {
                mode_id: 1, // pathogen: high W, 1D
                w_eff: 30.0,
                d_eff: 1.0,
                system_size: 100,
                energy: 0.0,
            },
        ];
        let profile = membrane_profile(&modes).unwrap();
        assert!(
            profile.selectivity > 0.5,
            "Gut lining should be selective: S = {}",
            profile.selectivity
        );
        assert!(
            profile.transmissions[0] > profile.transmissions[1],
            "Nutrient should pass more than pathogen"
        );
    }

    #[test]
    fn membrane_selectivity_breached_barrier() {
        let modes = vec![
            MembraneMode {
                mode_id: 0,
                w_eff: 0.1,
                d_eff: 3.0,
                system_size: 5,
                energy: 0.0,
            },
            MembraneMode {
                mode_id: 1,
                w_eff: 0.1,
                d_eff: 3.0,
                system_size: 5,
                energy: 0.0,
            },
        ];
        let profile = membrane_profile(&modes).unwrap();
        assert!(
            profile.selectivity < 0.01,
            "Breached barrier should be non-selective: S = {}",
            profile.selectivity
        );
    }

    #[test]
    fn membrane_empty_modes() {
        let profile = membrane_profile(&[]).unwrap();
        assert_eq!(profile.selectivity, 0.0);
        assert_eq!(profile.mean_transmission, 0.0);
    }

    #[test]
    fn membrane_invalid_disorder() {
        let mode = MembraneMode {
            mode_id: 0,
            w_eff: -1.0,
            d_eff: 3.0,
            system_size: 10,
            energy: 0.0,
        };
        assert!(membrane_permeability(&mode).is_err());
    }

    #[test]
    fn dimensional_xi_monotonic_in_dimension() {
        let xi_1d = dimensional_localization_length(5.0, 0.0, 1.0);
        let xi_2d = dimensional_localization_length(5.0, 0.0, 2.0);
        let xi_3d = dimensional_localization_length(5.0, 0.0, 3.0);
        assert!(
            xi_2d > xi_1d,
            "ξ should increase with dimension: ξ_2d={xi_2d}, ξ_1d={xi_1d}"
        );
        assert!(
            xi_3d > xi_2d,
            "ξ should increase with dimension: ξ_3d={xi_3d}, ξ_2d={xi_2d}"
        );
    }

    #[test]
    fn fractional_dimension_interpolates() {
        let xi_2 = dimensional_localization_length(5.0, 0.0, 2.0);
        let xi_25 = dimensional_localization_length(5.0, 0.0, 2.5);
        let xi_3 = dimensional_localization_length(5.0, 0.0, 3.0);
        assert!(
            xi_25 > xi_2 && xi_25 < xi_3,
            "Fractional d should interpolate: ξ_2={xi_2}, ξ_2.5={xi_25}, ξ_3={xi_3}"
        );
    }
}
