// SPDX-License-Identifier: AGPL-3.0-only

#![expect(clippy::needless_raw_string_hashes, reason = "WGSL readability")]

use super::super::params::{PHAGE_DEFENSE_N_PARAMS, PHAGE_DEFENSE_N_VARS, PhageDefenseParams};
use crate::numerical::ode_generic::OdeSystem;

/// Phage-bacteria defense arms race for `BatchedOdeRK4<PhageDefenseOde>`.
pub struct PhageDefenseOde;

impl OdeSystem for PhageDefenseOde {
    const N_VARS: usize = PHAGE_DEFENSE_N_VARS;
    const N_PARAMS: usize = PHAGE_DEFENSE_N_PARAMS;

    fn system_name() -> &'static str {
        "phage_defense"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn monod_d(r: f64, k: f64) -> f64 {
    return r / (k + r + 1e-30);
}
fn deriv(state: array<f64, 4>, params: array<f64, 11>, t: f64) -> array<f64, 4> {
    let mu_max = params[0];  let cost   = params[1];  let k_res  = params[2];
    let yld    = params[3];  let ads    = params[4];  let burst  = params[5];
    let eff    = params[6];  let decay  = params[7];  let inflow = params[8];
    let dilut  = params[9];  let death  = params[10];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let bd    = fmax_d(state[0], z); let bu    = fmax_d(state[1], z);
    let phage = fmax_d(state[2], z); let r     = fmax_d(state[3], z);

    let growth_limit = monod_d(r, k_res);
    let mu_d = mu_max * (one - cost) * growth_limit;
    let mu_u = mu_max * growth_limit;
    let inf_d = ads * bd * phage;
    let inf_u = ads * bu * phage;
    let kill_d = inf_d * (one - eff);

    var dy: array<f64, 4>;
    dy[0] = mu_d * bd - death * bd - kill_d;
    dy[1] = mu_u * bu - death * bu - inf_u;
    dy[2] = burst * (inf_u + kill_d) - ads * (bd + bu) * phage - decay * phage;
    dy[3] = inflow - yld * (mu_d * bd + mu_u * bu) - dilut * r;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = PhageDefenseParams::from_flat(params);
        let bd = state[0].max(0.0);
        let bu = state[1].max(0.0);
        let phage = state[2].max(0.0);
        let r = state[3].max(0.0);

        let growth_limit = r / (p.k_resource + r);
        let mu_d = p.mu_max * (1.0 - p.defense_cost) * growth_limit;
        let mu_u = p.mu_max * growth_limit;
        let infection_d = p.adsorption_rate * bd * phage;
        let infection_u = p.adsorption_rate * bu * phage;
        let kill_d = infection_d * (1.0 - p.defense_efficiency);

        let growth_defended = p.death_rate.mul_add(-bd, mu_d * bd - kill_d);
        let growth_undefended = p.death_rate.mul_add(-bu, mu_u * bu - infection_u);
        let burst_from_u = p.burst_size * infection_u;
        let burst_from_d = p.burst_size * (1.0 - p.defense_efficiency) * infection_d;
        let d_phage = (p.adsorption_rate * (bd + bu)).mul_add(
            -phage,
            p.phage_decay.mul_add(-phage, burst_from_u + burst_from_d),
        );
        let consumption = p.yield_coeff * (mu_d * bd + mu_u * bu);
        let d_r = p
            .resource_dilution
            .mul_add(-r, p.resource_inflow - consumption);

        vec![growth_defended, growth_undefended, d_phage, d_r]
    }
}
