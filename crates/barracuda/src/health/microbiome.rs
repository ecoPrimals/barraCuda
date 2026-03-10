// SPDX-License-Identifier: AGPL-3.0-only

//! Microbiome health primitives — SCFA production, antibiotic perturbation,
//! gut-brain serotonin axis.
//!
//! Absorbed from healthSpring V19 (Exp078–080, Exp084).

/// SCFA production parameters (Michaelis-Menten kinetics per metabolite).
#[derive(Debug, Clone, Copy)]
pub struct ScfaParams {
    /// Maximum acetate production rate (mmol/L/h).
    pub vmax_acetate: f64,
    /// Acetate Michaelis constant (g/L fiber).
    pub km_acetate: f64,
    /// Maximum propionate production rate (mmol/L/h).
    pub vmax_propionate: f64,
    /// Propionate Michaelis constant (g/L fiber).
    pub km_propionate: f64,
    /// Maximum butyrate production rate (mmol/L/h).
    pub vmax_butyrate: f64,
    /// Butyrate Michaelis constant (g/L fiber).
    pub km_butyrate: f64,
}

/// SCFA production result.
#[derive(Debug, Clone, Copy)]
pub struct ScfaResult {
    /// Acetate production rate (mmol/L/h).
    pub acetate: f64,
    /// Propionate production rate (mmol/L/h).
    pub propionate: f64,
    /// Butyrate production rate (mmol/L/h).
    pub butyrate: f64,
}

/// Healthy gut SCFA parameters (Cummings & Macfarlane 1991).
pub const SCFA_HEALTHY_PARAMS: ScfaParams = ScfaParams {
    vmax_acetate: 60.0,
    km_acetate: 5.0,
    vmax_propionate: 25.0,
    km_propionate: 4.0,
    vmax_butyrate: 20.0,
    km_butyrate: 3.5,
};

/// Dysbiotic gut SCFA parameters (reduced butyrate production).
pub const SCFA_DYSBIOTIC_PARAMS: ScfaParams = ScfaParams {
    vmax_acetate: 55.0,
    km_acetate: 6.0,
    vmax_propionate: 20.0,
    km_propionate: 5.0,
    vmax_butyrate: 8.0,
    km_butyrate: 5.0,
};

fn michaelis_menten(vmax: f64, km: f64, substrate: f64) -> f64 {
    vmax * substrate / (km + substrate)
}

/// Compute SCFA production for a given fiber substrate concentration.
#[must_use]
pub fn scfa_production(params: &ScfaParams, fiber: f64) -> ScfaResult {
    ScfaResult {
        acetate: michaelis_menten(params.vmax_acetate, params.km_acetate, fiber),
        propionate: michaelis_menten(params.vmax_propionate, params.km_propionate, fiber),
        butyrate: michaelis_menten(params.vmax_butyrate, params.km_butyrate, fiber),
    }
}

impl ScfaResult {
    /// Total SCFA output (acetate + propionate + butyrate).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.acetate + self.propionate + self.butyrate
    }

    /// Butyrate fraction (marker of gut health).
    #[must_use]
    pub fn butyrate_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            0.0
        } else {
            self.butyrate / total
        }
    }
}

/// Simulate antibiotic perturbation on a community.
///
/// Models exponential kill with species-specific susceptibility followed by
/// recovery. Returns the perturbed abundance vector.
///
/// `kill_rate` is the per-hour mortality (higher = more susceptible).
/// `duration_h` is the antibiotic exposure time.
///
/// # Panics
///
/// Panics if `abundances` and `susceptibilities` have different lengths.
#[must_use]
pub fn antibiotic_perturbation(
    abundances: &[f64],
    susceptibilities: &[f64],
    duration_h: f64,
) -> Vec<f64> {
    assert_eq!(abundances.len(), susceptibilities.len());
    abundances
        .iter()
        .zip(susceptibilities)
        .map(|(&a, &s)| (a * (-s * duration_h).exp()).max(0.0))
        .collect()
}

/// Gut serotonin production model.
///
/// ~95% of body serotonin is produced by enterochromaffin cells in the gut,
/// modulated by microbial tryptophan metabolism.
///
/// `tryptophan_mg` — dietary tryptophan intake.
/// `microbiome_factor` — 0.0 (germ-free) to 1.0 (healthy colonization).
///
/// Returns estimated serotonin production rate (arbitrary units).
#[must_use]
pub fn gut_serotonin_production(tryptophan_mg: f64, microbiome_factor: f64) -> f64 {
    let base_rate = 0.05 * tryptophan_mg;
    let microbial_boost = 1.0 + 0.6 * microbiome_factor;
    base_rate * microbial_boost
}

/// Tryptophan availability after microbial metabolism.
///
/// Gut bacteria consume tryptophan for indole production. Higher microbial
/// diversity → more tryptophan metabolism → less available for host serotonin.
#[must_use]
pub fn tryptophan_availability(intake_mg: f64, diversity_index: f64) -> f64 {
    let microbial_consumption = 0.15 * diversity_index.min(5.0);
    (intake_mg * (1.0 - microbial_consumption / 5.0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scfa_healthy() {
        let r = scfa_production(&SCFA_HEALTHY_PARAMS, 10.0);
        assert!(r.acetate > 0.0);
        assert!(r.propionate > 0.0);
        assert!(r.butyrate > 0.0);
        assert!(r.total() > 0.0);
    }

    #[test]
    fn test_scfa_dysbiotic_lower_butyrate() {
        let h = scfa_production(&SCFA_HEALTHY_PARAMS, 10.0);
        let d = scfa_production(&SCFA_DYSBIOTIC_PARAMS, 10.0);
        assert!(d.butyrate_fraction() < h.butyrate_fraction());
    }

    #[test]
    fn test_scfa_zero_fiber() {
        let r = scfa_production(&SCFA_HEALTHY_PARAMS, 0.0);
        assert!((r.total()).abs() < 1e-10);
    }

    #[test]
    fn test_antibiotic_perturbation() {
        let abundances = vec![1000.0, 500.0, 200.0];
        let susceptibilities = vec![0.1, 0.5, 0.01];
        let perturbed = antibiotic_perturbation(&abundances, &susceptibilities, 24.0);
        assert!(
            perturbed[1] < perturbed[0],
            "more susceptible species depleted more"
        );
        assert!(
            perturbed[2] > perturbed[1],
            "resistant species survives better"
        );
    }

    #[test]
    fn test_gut_serotonin() {
        let gf = gut_serotonin_production(100.0, 0.0);
        let colonized = gut_serotonin_production(100.0, 1.0);
        assert!(colonized > gf, "microbiome boosts serotonin");
    }

    #[test]
    fn test_tryptophan_availability() {
        let high_div = tryptophan_availability(100.0, 4.0);
        let low_div = tryptophan_availability(100.0, 1.0);
        assert!(high_div < low_div, "higher diversity → more consumption");
        assert!(tryptophan_availability(100.0, 0.0) > 99.0);
    }
}
