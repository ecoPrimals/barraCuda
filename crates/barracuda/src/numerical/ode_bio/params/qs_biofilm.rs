// SPDX-License-Identifier: AGPL-3.0-only
//! QS / c-di-GMP / biofilm model parameters (Waters 2008).

/// QS / c-di-GMP / biofilm model parameters (Waters 2008).
///
/// Used directly as the base for [`super::BistableParams`]. Not a standalone
/// `OdeSystem` in this module — the monostable version lives in wetSpring.
#[derive(Debug, Clone)]
pub struct QsBiofilmParams {
    /// Maximum specific growth rate (h⁻¹).
    pub mu_max: f64,
    /// Carrying capacity (cells/mL).
    pub k_cap: f64,
    /// Specific death rate (h⁻¹).
    pub death_rate: f64,
    /// Autoinducer production rate.
    pub k_ai_prod: f64,
    /// Autoinducer degradation rate (h⁻¹).
    pub d_ai: f64,
    /// Maximum `HapR` transcription rate.
    pub k_hapr_max: f64,
    /// Half-saturation for AI activation of `HapR`.
    pub k_hapr_ai: f64,
    /// Hill coefficient for `HapR` induction by AI.
    pub n_hapr: f64,
    /// `HapR` degradation rate (h⁻¹).
    pub d_hapr: f64,
    /// Basal diguanylate cyclase (DGC) activity.
    pub k_dgc_basal: f64,
    /// `HapR` repression strength on DGC.
    pub k_dgc_rep: f64,
    /// Basal phosphodiesterase (PDE) activity.
    pub k_pde_basal: f64,
    /// `HapR` activation strength on PDE.
    pub k_pde_act: f64,
    /// c-di-GMP degradation rate (h⁻¹).
    pub d_cdg: f64,
    /// Maximum biofilm production rate.
    pub k_bio_max: f64,
    /// Half-saturation for c-di-GMP activation of biofilm production.
    pub k_bio_cdg: f64,
    /// Hill coefficient for biofilm induction by c-di-GMP.
    pub n_bio: f64,
    /// Biofilm matrix degradation rate (h⁻¹).
    pub d_bio: f64,
}

/// Number of parameters in the QS biofilm model (flat buffer size).
pub const QS_BIOFILM_N_PARAMS: usize = 18;

impl Default for QsBiofilmParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_ai_prod: 5.0,
            d_ai: 1.0,
            k_hapr_max: 1.0,
            k_hapr_ai: 0.5,
            n_hapr: 2.0,
            d_hapr: 0.5,
            k_dgc_basal: 2.0,
            k_dgc_rep: 0.8,
            k_pde_basal: 0.5,
            k_pde_act: 2.0,
            d_cdg: 0.3,
            k_bio_max: 1.0,
            k_bio_cdg: 1.5,
            n_bio: 2.0,
            d_bio: 0.2,
        }
    }
}

impl QsBiofilmParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; QS_BIOFILM_N_PARAMS] {
        [
            self.mu_max,
            self.k_cap,
            self.death_rate,
            self.k_ai_prod,
            self.d_ai,
            self.k_hapr_max,
            self.k_hapr_ai,
            self.n_hapr,
            self.d_hapr,
            self.k_dgc_basal,
            self.k_dgc_rep,
            self.k_pde_basal,
            self.k_pde_act,
            self.d_cdg,
            self.k_bio_max,
            self.k_bio_cdg,
            self.n_bio,
            self.d_bio,
        ]
    }

    /// Reconstructs parameters from a flat array (e.g. from GPU buffer).
    ///
    /// # Panics
    /// Panics if `flat.len() < QS_BIOFILM_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= QS_BIOFILM_N_PARAMS,
            "need {QS_BIOFILM_N_PARAMS} values"
        );
        Self {
            mu_max: flat[0],
            k_cap: flat[1],
            death_rate: flat[2],
            k_ai_prod: flat[3],
            d_ai: flat[4],
            k_hapr_max: flat[5],
            k_hapr_ai: flat[6],
            n_hapr: flat[7],
            d_hapr: flat[8],
            k_dgc_basal: flat[9],
            k_dgc_rep: flat[10],
            k_pde_basal: flat[11],
            k_pde_act: flat[12],
            d_cdg: flat[13],
            k_bio_max: flat[14],
            k_bio_cdg: flat[15],
            n_bio: flat[16],
            d_bio: flat[17],
        }
    }
}
