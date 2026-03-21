// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cooperative QS game theory parameters (Bruger & Waters 2018).

/// Cooperative QS game theory: cooperators vs cheaters with frequency-dependent fitness.
#[derive(Debug, Clone)]
pub struct CooperationParams {
    /// Cooperator growth rate (h⁻¹).
    pub mu_coop: f64,
    /// Cheater growth rate (h⁻¹).
    pub mu_cheat: f64,
    /// Carrying capacity (cells/mL).
    pub k_cap: f64,
    /// Specific death rate (h⁻¹).
    pub death_rate: f64,
    /// Autoinducer production rate.
    pub k_ai_prod: f64,
    /// Autoinducer degradation rate (h⁻¹).
    pub d_ai: f64,
    /// Maximum benefit from cooperation.
    pub benefit: f64,
    /// Half-saturation for benefit response.
    pub k_benefit: f64,
    /// Cost of cooperation.
    pub cost: f64,
    /// Basal biofilm production rate.
    pub k_bio: f64,
    /// Half-saturation for AI activation of biofilm.
    pub k_bio_ai: f64,
    /// Dispersal bonus for biofilm producers.
    pub dispersal_bonus: f64,
    /// Biofilm degradation rate (h⁻¹).
    pub d_bio: f64,
}

/// Number of state variables in the cooperation model.
pub const COOPERATION_N_VARS: usize = 4;
/// Number of parameters in the cooperation model (flat buffer size).
pub const COOPERATION_N_PARAMS: usize = 13;

impl Default for CooperationParams {
    fn default() -> Self {
        Self {
            mu_coop: 0.7,
            mu_cheat: 0.75,
            k_cap: 1.0,
            death_rate: 0.02,
            k_ai_prod: 5.0,
            d_ai: 1.0,
            benefit: 0.3,
            k_benefit: 0.5,
            cost: 0.05,
            k_bio: 1.0,
            k_bio_ai: 0.5,
            dispersal_bonus: 0.2,
            d_bio: 0.3,
        }
    }
}

impl CooperationParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; COOPERATION_N_PARAMS] {
        [
            self.mu_coop,
            self.mu_cheat,
            self.k_cap,
            self.death_rate,
            self.k_ai_prod,
            self.d_ai,
            self.benefit,
            self.k_benefit,
            self.cost,
            self.k_bio,
            self.k_bio_ai,
            self.dispersal_bonus,
            self.d_bio,
        ]
    }

    /// Reconstructs parameters from a flat array (e.g. from GPU buffer).
    ///
    /// # Panics
    /// Panics if `flat.len() < COOPERATION_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= COOPERATION_N_PARAMS,
            "need {COOPERATION_N_PARAMS} values"
        );
        Self {
            mu_coop: flat[0],
            mu_cheat: flat[1],
            k_cap: flat[2],
            death_rate: flat[3],
            k_ai_prod: flat[4],
            d_ai: flat[5],
            benefit: flat[6],
            k_benefit: flat[7],
            cost: flat[8],
            k_bio: flat[9],
            k_bio_ai: flat[10],
            dispersal_bonus: flat[11],
            d_bio: flat[12],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let d = CooperationParams::default();
        let r = CooperationParams::from_flat(&d.to_flat());
        assert!(f64::abs(d.mu_coop - r.mu_coop) < 1e-15);
        assert!(f64::abs(d.mu_cheat - r.mu_cheat) < 1e-15);
        assert!(f64::abs(d.k_cap - r.k_cap) < 1e-15);
        assert!(f64::abs(d.death_rate - r.death_rate) < 1e-15);
        assert!(f64::abs(d.k_ai_prod - r.k_ai_prod) < 1e-15);
        assert!(f64::abs(d.d_ai - r.d_ai) < 1e-15);
        assert!(f64::abs(d.benefit - r.benefit) < 1e-15);
        assert!(f64::abs(d.k_benefit - r.k_benefit) < 1e-15);
        assert!(f64::abs(d.cost - r.cost) < 1e-15);
        assert!(f64::abs(d.k_bio - r.k_bio) < 1e-15);
        assert!(f64::abs(d.k_bio_ai - r.k_bio_ai) < 1e-15);
        assert!(f64::abs(d.dispersal_bonus - r.dispersal_bonus) < 1e-15);
        assert!(f64::abs(d.d_bio - r.d_bio) < 1e-15);
    }

    #[test]
    fn test_flat_length() {
        assert_eq!(
            CooperationParams::default().to_flat().len(),
            COOPERATION_N_PARAMS
        );
    }
}
