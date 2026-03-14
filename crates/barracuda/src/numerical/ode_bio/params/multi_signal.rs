// SPDX-License-Identifier: AGPL-3.0-only
//! Dual-signal QS regulatory network parameters (Srivastava et al. 2011).

/// Dual-signal QS regulatory network: CAI-1 + AI-2 converge on `HapR`.
#[derive(Debug, Clone)]
pub struct MultiSignalParams {
    /// Maximum specific growth rate (h⁻¹).
    pub mu_max: f64,
    /// Carrying capacity (cells/mL).
    pub k_cap: f64,
    /// Specific death rate (h⁻¹).
    pub death_rate: f64,
    /// CAI-1 autoinducer production rate.
    pub k_cai1_prod: f64,
    /// CAI-1 degradation rate (h⁻¹).
    pub d_cai1: f64,
    /// `CqsS` receptor sensitivity (CAI-1 sensing).
    pub k_cqs: f64,
    /// AI-2 autoinducer production rate.
    pub k_ai2_prod: f64,
    /// AI-2 degradation rate (h⁻¹).
    pub d_ai2: f64,
    /// `LuxPQ` receptor sensitivity (AI-2 sensing).
    pub k_luxpq: f64,
    /// `LuxO` phosphorylation rate.
    pub k_luxo_phos: f64,
    /// Phosphorylated `LuxO` degradation rate (h⁻¹).
    pub d_luxo_p: f64,
    /// Maximum `HapR` transcription rate.
    pub k_hapr_max: f64,
    /// Hill coefficient for `HapR` repression.
    pub n_repress: f64,
    /// Half-saturation for LuxO-P repression of `HapR`.
    pub k_repress: f64,
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

/// Number of state variables in the multi-signal model.
pub const MULTI_SIGNAL_N_VARS: usize = 7;
/// Number of parameters in the multi-signal model (flat buffer size).
pub const MULTI_SIGNAL_N_PARAMS: usize = 24;

impl Default for MultiSignalParams {
    fn default() -> Self {
        Self {
            mu_max: 0.8,
            k_cap: 1.0,
            death_rate: 0.02,
            k_cai1_prod: 3.0,
            d_cai1: 1.0,
            k_cqs: 0.5,
            k_ai2_prod: 3.0,
            d_ai2: 1.0,
            k_luxpq: 0.5,
            k_luxo_phos: 2.0,
            d_luxo_p: 0.5,
            k_hapr_max: 1.0,
            n_repress: 2.0,
            k_repress: 0.5,
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

impl MultiSignalParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub const fn to_flat(&self) -> [f64; MULTI_SIGNAL_N_PARAMS] {
        [
            self.mu_max,
            self.k_cap,
            self.death_rate,
            self.k_cai1_prod,
            self.d_cai1,
            self.k_cqs,
            self.k_ai2_prod,
            self.d_ai2,
            self.k_luxpq,
            self.k_luxo_phos,
            self.d_luxo_p,
            self.k_hapr_max,
            self.n_repress,
            self.k_repress,
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
    /// Panics if `flat.len() < MULTI_SIGNAL_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= MULTI_SIGNAL_N_PARAMS,
            "need {MULTI_SIGNAL_N_PARAMS} values"
        );
        Self {
            mu_max: flat[0],
            k_cap: flat[1],
            death_rate: flat[2],
            k_cai1_prod: flat[3],
            d_cai1: flat[4],
            k_cqs: flat[5],
            k_ai2_prod: flat[6],
            d_ai2: flat[7],
            k_luxpq: flat[8],
            k_luxo_phos: flat[9],
            d_luxo_p: flat[10],
            k_hapr_max: flat[11],
            n_repress: flat[12],
            k_repress: flat[13],
            d_hapr: flat[14],
            k_dgc_basal: flat[15],
            k_dgc_rep: flat[16],
            k_pde_basal: flat[17],
            k_pde_act: flat[18],
            d_cdg: flat[19],
            k_bio_max: flat[20],
            k_bio_cdg: flat[21],
            n_bio: flat[22],
            d_bio: flat[23],
        }
    }
}
