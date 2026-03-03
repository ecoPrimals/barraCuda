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
    pub mu_max: f64,
    pub k_cap: f64,
    pub death_rate: f64,
    pub k_ai_prod: f64,
    pub d_ai: f64,
    pub k_hapr_max: f64,
    pub k_hapr_ai: f64,
    pub n_hapr: f64,
    pub d_hapr: f64,
    pub k_dgc_basal: f64,
    pub k_dgc_rep: f64,
    pub k_pde_basal: f64,
    pub k_pde_act: f64,
    pub d_cdg: f64,
    pub k_bio_max: f64,
    pub k_bio_cdg: f64,
    pub n_bio: f64,
    pub d_bio: f64,
}

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
    pub mu_max: f64,
    pub k_cap: f64,
    pub death_rate: f64,
    pub k_cdg_prod: f64,
    pub d_cdg: f64,
    pub k_vpsr_charge: f64,
    pub k_vpsr_discharge: f64,
    pub n_vpsr: f64,
    pub k_vpsr_cdg: f64,
    pub w_biofilm: f64,
    pub w_motility: f64,
    pub w_rugose: f64,
    pub d_bio: f64,
    pub d_mot: f64,
    pub d_rug: f64,
    pub stress_factor: f64,
}

pub const CAPACITOR_N_VARS: usize = 6;
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
    pub mu_coop: f64,
    pub mu_cheat: f64,
    pub k_cap: f64,
    pub death_rate: f64,
    pub k_ai_prod: f64,
    pub d_ai: f64,
    pub benefit: f64,
    pub k_benefit: f64,
    pub cost: f64,
    pub k_bio: f64,
    pub k_bio_ai: f64,
    pub dispersal_bonus: f64,
    pub d_bio: f64,
}

pub const COOPERATION_N_VARS: usize = 4;
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
    pub mu_max: f64,
    pub k_cap: f64,
    pub death_rate: f64,
    pub k_cai1_prod: f64,
    pub d_cai1: f64,
    pub k_cqs: f64,
    pub k_ai2_prod: f64,
    pub d_ai2: f64,
    pub k_luxpq: f64,
    pub k_luxo_phos: f64,
    pub d_luxo_p: f64,
    pub k_hapr_max: f64,
    pub n_repress: f64,
    pub k_repress: f64,
    pub d_hapr: f64,
    pub k_dgc_basal: f64,
    pub k_dgc_rep: f64,
    pub k_pde_basal: f64,
    pub k_pde_act: f64,
    pub d_cdg: f64,
    pub k_bio_max: f64,
    pub k_bio_cdg: f64,
    pub n_bio: f64,
    pub d_bio: f64,
}

pub const MULTI_SIGNAL_N_VARS: usize = 7;
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
    pub base: QsBiofilmParams,
    pub alpha_fb: f64,
    pub n_fb: f64,
    pub k_fb: f64,
}

pub const BISTABLE_N_VARS: usize = 5;
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
    pub mu_max: f64,
    pub defense_cost: f64,
    pub k_resource: f64,
    pub yield_coeff: f64,
    pub adsorption_rate: f64,
    pub burst_size: f64,
    pub defense_efficiency: f64,
    pub phage_decay: f64,
    pub resource_inflow: f64,
    pub resource_dilution: f64,
    pub death_rate: f64,
}

pub const PHAGE_DEFENSE_N_VARS: usize = 4;
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
