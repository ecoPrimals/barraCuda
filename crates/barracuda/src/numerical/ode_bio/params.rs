// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parameter structs for the 5 biological ODE systems.
//!
//! Each struct carries the kinetic constants for one model and provides
//! `to_flat()` / `from_flat()` for GPU buffer packing. The flat layout
//! matches the `params` array indices in the WGSL derivative functions.

// ── QsBiofilm (base for Bistable) ────────────────────────────────────────────

/// QS / c-di-GMP / biofilm model parameters (Waters 2008).
///
/// Used directly as the base for [`BistableParams`]. Not a standalone
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
    /// Maximum HapR transcription rate.
    pub k_hapr_max: f64,
    /// Half-saturation for AI activation of HapR.
    pub k_hapr_ai: f64,
    /// Hill coefficient for HapR induction by AI.
    pub n_hapr: f64,
    /// HapR degradation rate (h⁻¹).
    pub d_hapr: f64,
    /// Basal diguanylate cyclase (DGC) activity.
    pub k_dgc_basal: f64,
    /// HapR repression strength on DGC.
    pub k_dgc_rep: f64,
    /// Basal phosphodiesterase (PDE) activity.
    pub k_pde_basal: f64,
    /// HapR activation strength on PDE.
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

// ── Capacitor (Mhatre et al. 2020) ──────────────────────────────────────────

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
    /// VpsR charge rate (c-di-GMP uptake into capacitor).
    pub k_vpsr_charge: f64,
    /// VpsR discharge rate (release from capacitor).
    pub k_vpsr_discharge: f64,
    /// Hill coefficient for VpsR response.
    pub n_vpsr: f64,
    /// Half-saturation for c-di-GMP activation of VpsR.
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

// ── Cooperation (Bruger & Waters 2018) ───────────────────────────────────────

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

// ── MultiSignal (Srivastava et al. 2011) ────────────────────────────────────

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
    /// CqsS receptor sensitivity (CAI-1 sensing).
    pub k_cqs: f64,
    /// AI-2 autoinducer production rate.
    pub k_ai2_prod: f64,
    /// AI-2 degradation rate (h⁻¹).
    pub d_ai2: f64,
    /// LuxPQ receptor sensitivity (AI-2 sensing).
    pub k_luxpq: f64,
    /// LuxO phosphorylation rate.
    pub k_luxo_phos: f64,
    /// Phosphorylated LuxO degradation rate (h⁻¹).
    pub d_luxo_p: f64,
    /// Maximum HapR transcription rate.
    pub k_hapr_max: f64,
    /// Hill coefficient for HapR repression.
    pub n_repress: f64,
    /// Half-saturation for LuxO-P repression of HapR.
    pub k_repress: f64,
    /// HapR degradation rate (h⁻¹).
    pub d_hapr: f64,
    /// Basal diguanylate cyclase (DGC) activity.
    pub k_dgc_basal: f64,
    /// HapR repression strength on DGC.
    pub k_dgc_rep: f64,
    /// Basal phosphodiesterase (PDE) activity.
    pub k_pde_basal: f64,
    /// HapR activation strength on PDE.
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

// ── Bistable (Fernandez et al. 2020) ────────────────────────────────────────

/// Bistable phenotypic switching: positive feedback on c-di-GMP production
/// from biofilm state creates hysteresis.
#[derive(Debug, Clone)]
pub struct BistableParams {
    /// Base QS biofilm parameters (shared with monostable model).
    pub base: QsBiofilmParams,
    /// Feedback strength: biofilm → c-di-GMP production amplification.
    pub alpha_fb: f64,
    /// Hill coefficient for positive feedback.
    pub n_fb: f64,
    /// Half-saturation for feedback activation by biofilm state.
    pub k_fb: f64,
}

/// Number of state variables in the bistable model.
pub const BISTABLE_N_VARS: usize = 5;
/// Number of parameters in the bistable model (flat buffer size).
pub const BISTABLE_N_PARAMS: usize = 21;

impl Default for BistableParams {
    fn default() -> Self {
        let base = QsBiofilmParams {
            k_dgc_rep: 0.3,
            k_pde_act: 0.5,
            k_bio_cdg: 1.5,
            n_bio: 4.0,
            ..QsBiofilmParams::default()
        };
        Self {
            base,
            alpha_fb: 3.0,
            n_fb: 4.0,
            k_fb: 0.6,
        }
    }
}

impl BistableParams {
    /// Packs parameters into a flat array for GPU buffer upload.
    #[must_use]
    pub fn to_flat(&self) -> [f64; BISTABLE_N_PARAMS] {
        let base = self.base.to_flat();
        let mut out = [0.0; BISTABLE_N_PARAMS];
        out[..QS_BIOFILM_N_PARAMS].copy_from_slice(&base);
        out[18] = self.alpha_fb;
        out[19] = self.n_fb;
        out[20] = self.k_fb;
        out
    }

    /// Reconstructs parameters from a flat array (e.g. from GPU buffer).
    ///
    /// # Panics
    /// Panics if `flat.len() < BISTABLE_N_PARAMS`.
    #[must_use]
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len() >= BISTABLE_N_PARAMS,
            "need {BISTABLE_N_PARAMS} values"
        );
        Self {
            base: QsBiofilmParams::from_flat(flat),
            alpha_fb: flat[18],
            n_fb: flat[19],
            k_fb: flat[20],
        }
    }
}

// ── PhageDefense (Hsueh, Severin et al. 2022) ───────────────────────────────

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
