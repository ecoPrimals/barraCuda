// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phage-bacteria defense arms race parameters (Hsueh, Severin et al. 2022).

/// Phage-bacteria defense arms race: `DCD` deaminase reduces phage burst size
/// at a growth cost.
#[derive(Debug, Clone)]
pub struct PhageDefenseParams {
    /// Maximum specific growth rate (h⁻¹).
    pub mu_max: f64,
    /// Growth cost of DCD deaminase defense (fractional reduction).
    pub defense_cost: f64,
    /// Half-saturation for resource-limited growth.
    pub k_resource: f64,
    /// Yield coefficient (biomass per unit resource consumed).
    pub yield_coeff: f64,
    /// Phage adsorption rate (mL·h⁻¹ per cell).
    pub adsorption_rate: f64,
    /// Phage burst size (virions per infected cell).
    pub burst_size: f64,
    /// Fraction of phage inactivated by DCD defense.
    pub defense_efficiency: f64,
    /// Phage decay rate (h⁻¹).
    pub phage_decay: f64,
    /// Resource inflow rate (chemostat).
    pub resource_inflow: f64,
    /// Resource dilution rate (h⁻¹).
    pub resource_dilution: f64,
    /// Specific death rate (h⁻¹).
    pub death_rate: f64,
}

/// Number of state variables in the phage defense model.
pub const PHAGE_DEFENSE_N_VARS: usize = 4;
/// Number of parameters in the phage defense model (flat buffer size).
pub const PHAGE_DEFENSE_N_PARAMS: usize = 11;

impl Default for PhageDefenseParams {
    fn default() -> Self {
        Self {
            mu_max: 1.0,
            defense_cost: 0.15,
            k_resource: 0.5,
            yield_coeff: 0.5,
            adsorption_rate: 1e-7,
            burst_size: 50.0,
            defense_efficiency: 0.9,
            phage_decay: 0.1,
            resource_inflow: 10.0,
            resource_dilution: 0.1,
            death_rate: 0.05,
        }
    }
}

impl PhageDefenseParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; PHAGE_DEFENSE_N_PARAMS] {
        [
            self.mu_max,
            self.defense_cost,
            self.k_resource,
            self.yield_coeff,
            self.adsorption_rate,
            self.burst_size,
            self.defense_efficiency,
            self.phage_decay,
            self.resource_inflow,
            self.resource_dilution,
            self.death_rate,
        ]
    }

    /// Reconstructs parameters from a flat array (e.g. from GPU buffer).
    ///
    /// # Panics
    /// Panics if `flat.len() < PHAGE_DEFENSE_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= PHAGE_DEFENSE_N_PARAMS,
            "need {PHAGE_DEFENSE_N_PARAMS} values"
        );
        Self {
            mu_max: flat[0],
            defense_cost: flat[1],
            k_resource: flat[2],
            yield_coeff: flat[3],
            adsorption_rate: flat[4],
            burst_size: flat[5],
            defense_efficiency: flat[6],
            phage_decay: flat[7],
            resource_inflow: flat[8],
            resource_dilution: flat[9],
            death_rate: flat[10],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let d = PhageDefenseParams::default();
        let r = PhageDefenseParams::from_flat(&d.to_flat());
        assert!(f64::abs(d.mu_max - r.mu_max) < 1e-15);
        assert!(f64::abs(d.defense_cost - r.defense_cost) < 1e-15);
        assert!(f64::abs(d.k_resource - r.k_resource) < 1e-15);
        assert!(f64::abs(d.yield_coeff - r.yield_coeff) < 1e-15);
        assert!(f64::abs(d.adsorption_rate - r.adsorption_rate) < 1e-15);
        assert!(f64::abs(d.burst_size - r.burst_size) < 1e-15);
        assert!(f64::abs(d.defense_efficiency - r.defense_efficiency) < 1e-15);
        assert!(f64::abs(d.phage_decay - r.phage_decay) < 1e-15);
        assert!(f64::abs(d.resource_inflow - r.resource_inflow) < 1e-15);
        assert!(f64::abs(d.resource_dilution - r.resource_dilution) < 1e-15);
        assert!(f64::abs(d.death_rate - r.death_rate) < 1e-15);
    }

    #[test]
    fn test_flat_length() {
        assert_eq!(
            PhageDefenseParams::default().to_flat().len(),
            PHAGE_DEFENSE_N_PARAMS
        );
    }
}
