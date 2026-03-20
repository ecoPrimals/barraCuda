// SPDX-License-Identifier: AGPL-3.0-or-later

//! Michaelis-Menten pharmacokinetics — nonlinear PK simulation and analysis.
//!
//! Absorbed from healthSpring V19 `pkpd/nonlinear.rs` (Exp077, Exp084).

/// Michaelis-Menten PK parameters.
#[derive(Debug, Clone, Copy)]
pub struct MichaelisMentenParams {
    /// Maximum elimination rate (mg/L/h).
    pub vmax: f64,
    /// Michaelis constant (mg/L).
    pub km: f64,
    /// Volume of distribution (L).
    pub vd: f64,
}

/// Phenytoin reference parameters (Rowland & Tozer).
pub const PHENYTOIN_PARAMS: MichaelisMentenParams = MichaelisMentenParams {
    vmax: 7.26,
    km: 4.4,
    vd: 50.0,
};

/// Euler-method simulation of Michaelis-Menten elimination.
///
/// Returns concentration time-course `[C(0), C(dt), C(2*dt), ...]`.
#[must_use]
pub fn mm_pk_simulate(
    params: &MichaelisMentenParams,
    c0: f64,
    dt: f64,
    n_steps: usize,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(n_steps + 1);
    let mut c = c0;
    result.push(c);
    for _ in 0..n_steps {
        let elim = params.vmax * c / (params.km + c);
        c = (c - (elim / params.vd) * dt).max(0.0);
        result.push(c);
    }
    result
}

/// Steady-state concentration during constant-rate infusion.
///
/// At steady state: `R_inf = Vmax * Css / (Km + Css)` → `Css = Km * R_inf / (Vmax - R_inf)`.
/// Returns `None` if infusion rate >= Vmax (no steady state possible).
#[must_use]
pub fn mm_css_infusion(params: &MichaelisMentenParams, infusion_rate: f64) -> Option<f64> {
    if infusion_rate >= params.vmax {
        return None;
    }
    Some(params.km * infusion_rate / (params.vmax - infusion_rate))
}

/// Apparent half-life at a given concentration.
///
/// For Michaelis-Menten: `t½ ≈ 0.693 * Vd * (Km + C) / Vmax`.
#[must_use]
pub fn mm_apparent_half_life(params: &MichaelisMentenParams, concentration: f64) -> f64 {
    0.693 * params.vd * (params.km + concentration) / params.vmax
}

/// Trapezoidal AUC from a concentration time-course.
#[must_use]
pub fn mm_auc(concentrations: &[f64], dt: f64) -> f64 {
    if concentrations.len() < 2 {
        return 0.0;
    }
    concentrations
        .windows(2)
        .map(|w| (w[0] + w[1]) * 0.5 * dt)
        .sum()
}

/// Analytical AUC for Michaelis-Menten from `c0` to `c_final`.
///
/// `AUC = Vd/Vmax * [Km * ln(C0/Cf) + (C0 - Cf)]`.
#[must_use]
pub fn mm_auc_analytical(params: &MichaelisMentenParams, c0: f64, c_final: f64) -> f64 {
    if c_final <= 0.0 || c0 <= 0.0 {
        return 0.0;
    }
    params.vd / params.vmax * (params.km * (c0 / c_final).ln() + (c0 - c_final))
}

/// Nonlinearity ratio: `C / (Km + C)`. Measures departure from first-order kinetics.
///
/// - Near 0: first-order (linear) regime
/// - Near 1: saturated (zero-order) regime
#[must_use]
pub fn mm_nonlinearity_ratio(params: &MichaelisMentenParams, concentration: f64) -> f64 {
    concentration / (params.km + concentration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phenytoin_simulation() {
        let c = mm_pk_simulate(&PHENYTOIN_PARAMS, 6.0, 0.1, 100);
        assert_eq!(c.len(), 101);
        assert!((c[0] - 6.0).abs() < 1e-10);
        assert!(c.last().copied().unwrap_or(0.0) < 6.0);
        for &ci in &c {
            assert!(ci >= 0.0, "concentration must be non-negative");
        }
    }

    #[test]
    fn test_css_infusion() {
        let css = mm_css_infusion(&PHENYTOIN_PARAMS, 3.0).unwrap();
        assert!(css > 0.0);
        assert!(mm_css_infusion(&PHENYTOIN_PARAMS, 10.0).is_none());
    }

    #[test]
    fn test_apparent_half_life() {
        let t_half_low = mm_apparent_half_life(&PHENYTOIN_PARAMS, 1.0);
        let t_half_high = mm_apparent_half_life(&PHENYTOIN_PARAMS, 20.0);
        assert!(
            t_half_high > t_half_low,
            "half-life increases with concentration"
        );
    }

    #[test]
    fn test_auc_trapezoidal() {
        let conc = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let auc = mm_auc(&conc, 1.0);
        assert!((auc - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_analytical() {
        let auc = mm_auc_analytical(&PHENYTOIN_PARAMS, 6.0, 3.0);
        assert!(auc > 0.0);
    }

    #[test]
    fn test_nonlinearity_ratio() {
        let low = mm_nonlinearity_ratio(&PHENYTOIN_PARAMS, 0.1);
        let high = mm_nonlinearity_ratio(&PHENYTOIN_PARAMS, 100.0);
        assert!(low < 0.1);
        assert!(high > 0.9);
    }
}
