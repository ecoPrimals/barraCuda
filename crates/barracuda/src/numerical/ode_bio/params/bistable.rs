// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bistable phenotypic switching parameters (Fernandez et al. 2020).

use super::qs_biofilm::{QS_BIOFILM_N_PARAMS, QsBiofilmParams};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_roundtrip() {
        let d = BistableParams::default();
        let r = BistableParams::from_flat(&d.to_flat());
        let db = &d.base;
        let rb = &r.base;
        assert!(f64::abs(db.mu_max - rb.mu_max) < 1e-15);
        assert!(f64::abs(db.k_cap - rb.k_cap) < 1e-15);
        assert!(f64::abs(db.death_rate - rb.death_rate) < 1e-15);
        assert!(f64::abs(db.k_ai_prod - rb.k_ai_prod) < 1e-15);
        assert!(f64::abs(db.d_ai - rb.d_ai) < 1e-15);
        assert!(f64::abs(db.k_hapr_max - rb.k_hapr_max) < 1e-15);
        assert!(f64::abs(db.k_hapr_ai - rb.k_hapr_ai) < 1e-15);
        assert!(f64::abs(db.n_hapr - rb.n_hapr) < 1e-15);
        assert!(f64::abs(db.d_hapr - rb.d_hapr) < 1e-15);
        assert!(f64::abs(db.k_dgc_basal - rb.k_dgc_basal) < 1e-15);
        assert!(f64::abs(db.k_dgc_rep - rb.k_dgc_rep) < 1e-15);
        assert!(f64::abs(db.k_pde_basal - rb.k_pde_basal) < 1e-15);
        assert!(f64::abs(db.k_pde_act - rb.k_pde_act) < 1e-15);
        assert!(f64::abs(db.d_cdg - rb.d_cdg) < 1e-15);
        assert!(f64::abs(db.k_bio_max - rb.k_bio_max) < 1e-15);
        assert!(f64::abs(db.k_bio_cdg - rb.k_bio_cdg) < 1e-15);
        assert!(f64::abs(db.n_bio - rb.n_bio) < 1e-15);
        assert!(f64::abs(db.d_bio - rb.d_bio) < 1e-15);
        assert!(f64::abs(d.alpha_fb - r.alpha_fb) < 1e-15);
        assert!(f64::abs(d.n_fb - r.n_fb) < 1e-15);
        assert!(f64::abs(d.k_fb - r.k_fb) < 1e-15);
    }

    #[test]
    fn test_flat_length() {
        assert_eq!(BistableParams::default().to_flat().len(), BISTABLE_N_PARAMS);
    }
}
