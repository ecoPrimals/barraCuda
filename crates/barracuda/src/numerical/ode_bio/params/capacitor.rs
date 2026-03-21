// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phenotypic capacitor model parameters (Mhatre et al. 2020).

/// Phenotypic capacitor model — `VpsR` integrates c-di-GMP and distributes
/// its output to biofilm, motility, and rugose phenotypic channels.
#[derive(Debug, Clone)]
pub struct CapacitorParams {
    /// Maximum specific growth rate (h⁻¹).
    pub mu_max: f64,
    /// Carrying capacity (cells/mL).
    pub k_cap: f64,
    /// Specific death rate (h⁻¹).
    pub death_rate: f64,
    /// c-di-GMP production rate.
    pub k_cdg_prod: f64,
    /// c-di-GMP degradation rate (h⁻¹).
    pub d_cdg: f64,
    /// `VpsR` charge rate (c-di-GMP uptake into capacitor).
    pub k_vpsr_charge: f64,
    /// `VpsR` discharge rate (release from capacitor).
    pub k_vpsr_discharge: f64,
    /// Hill coefficient for `VpsR` response.
    pub n_vpsr: f64,
    /// Half-saturation for c-di-GMP activation of `VpsR`.
    pub k_vpsr_cdg: f64,
    /// Weight of biofilm channel in phenotypic distribution.
    pub w_biofilm: f64,
    /// Weight of motility channel in phenotypic distribution.
    pub w_motility: f64,
    /// Weight of rugose channel in phenotypic distribution.
    pub w_rugose: f64,
    /// Biofilm phenotype decay rate (h⁻¹).
    pub d_bio: f64,
    /// Motility phenotype decay rate (h⁻¹).
    pub d_mot: f64,
    /// Rugose phenotype decay rate (h⁻¹).
    pub d_rug: f64,
    /// Stress-induced amplification factor for phenotypic switching.
    pub stress_factor: f64,
}

/// Number of state variables in the capacitor model.
pub const CAPACITOR_N_VARS: usize = 6;
/// Number of parameters in the capacitor model (flat buffer size).
pub const CAPACITOR_N_PARAMS: usize = 16;

impl Default for CapacitorParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_cdg_prod: 2.0,
            d_cdg: 0.5,
            k_vpsr_charge: 1.0,
            k_vpsr_discharge: 0.3,
            n_vpsr: 3.0,
            k_vpsr_cdg: 1.0,
            w_biofilm: 0.8,
            w_motility: 0.6,
            w_rugose: 0.4,
            d_bio: 0.3,
            d_mot: 0.3,
            d_rug: 0.3,
            stress_factor: 1.0,
        }
    }
}

impl CapacitorParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; CAPACITOR_N_PARAMS] {
        [
            self.mu_max,
            self.k_cap,
            self.death_rate,
            self.k_cdg_prod,
            self.d_cdg,
            self.k_vpsr_charge,
            self.k_vpsr_discharge,
            self.n_vpsr,
            self.k_vpsr_cdg,
            self.w_biofilm,
            self.w_motility,
            self.w_rugose,
            self.d_bio,
            self.d_mot,
            self.d_rug,
            self.stress_factor,
        ]
    }

    /// Reconstructs parameters from a flat array (e.g. from GPU buffer).
    ///
    /// # Panics
    /// Panics if `flat.len() < CAPACITOR_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= CAPACITOR_N_PARAMS,
            "need {CAPACITOR_N_PARAMS} values"
        );
        Self {
            mu_max: flat[0],
            k_cap: flat[1],
            death_rate: flat[2],
            k_cdg_prod: flat[3],
            d_cdg: flat[4],
            k_vpsr_charge: flat[5],
            k_vpsr_discharge: flat[6],
            n_vpsr: flat[7],
            k_vpsr_cdg: flat[8],
            w_biofilm: flat[9],
            w_motility: flat[10],
            w_rugose: flat[11],
            d_bio: flat[12],
            d_mot: flat[13],
            d_rug: flat[14],
            stress_factor: flat[15],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let d = CapacitorParams::default();
        let r = CapacitorParams::from_flat(&d.to_flat());
        assert!(f64::abs(d.mu_max - r.mu_max) < 1e-15);
        assert!(f64::abs(d.k_cap - r.k_cap) < 1e-15);
        assert!(f64::abs(d.death_rate - r.death_rate) < 1e-15);
        assert!(f64::abs(d.k_cdg_prod - r.k_cdg_prod) < 1e-15);
        assert!(f64::abs(d.d_cdg - r.d_cdg) < 1e-15);
        assert!(f64::abs(d.k_vpsr_charge - r.k_vpsr_charge) < 1e-15);
        assert!(f64::abs(d.k_vpsr_discharge - r.k_vpsr_discharge) < 1e-15);
        assert!(f64::abs(d.n_vpsr - r.n_vpsr) < 1e-15);
        assert!(f64::abs(d.k_vpsr_cdg - r.k_vpsr_cdg) < 1e-15);
        assert!(f64::abs(d.w_biofilm - r.w_biofilm) < 1e-15);
        assert!(f64::abs(d.w_motility - r.w_motility) < 1e-15);
        assert!(f64::abs(d.w_rugose - r.w_rugose) < 1e-15);
        assert!(f64::abs(d.d_bio - r.d_bio) < 1e-15);
        assert!(f64::abs(d.d_mot - r.d_mot) < 1e-15);
        assert!(f64::abs(d.d_rug - r.d_rug) < 1e-15);
        assert!(f64::abs(d.stress_factor - r.stress_factor) < 1e-15);
    }

    #[test]
    fn test_flat_length() {
        assert_eq!(
            CapacitorParams::default().to_flat().len(),
            CAPACITOR_N_PARAMS
        );
    }
}
