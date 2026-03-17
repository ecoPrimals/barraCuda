// SPDX-License-Identifier: AGPL-3.0-only

#![expect(clippy::needless_raw_string_hashes, reason = "WGSL readability")]

use super::super::params::{CAPACITOR_N_PARAMS, CAPACITOR_N_VARS, CapacitorParams};
use crate::numerical::ode_generic::OdeSystem;

/// Phenotypic capacitor ODE for `BatchedOdeRK4<CapacitorOde>`.
pub struct CapacitorOde;

impl OdeSystem for CapacitorOde {
    const N_VARS: usize = CAPACITOR_N_VARS;
    const N_PARAMS: usize = CAPACITOR_N_PARAMS;

    fn system_name() -> &'static str {
        "capacitor"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fpow_d(base: f64, exp_val: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(exp_val * log(base));
}
fn hill_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let xn = fpow_d(x, n);
    let kn = fpow_d(k, n);
    return xn / (kn + xn + 1e-30);
}
fn deriv(state: array<f64, 6>, params: array<f64, 16>, t: f64) -> array<f64, 6> {
    let mu_max   = params[0];  let k_cap   = params[1];  let death    = params[2];
    let k_cdg    = params[3];  let d_cdg   = params[4];  let k_charge = params[5];
    let k_disch  = params[6];  let n_vpsr  = params[7];  let k_vcdg   = params[8];
    let w_bio    = params[9];  let w_mot   = params[10]; let w_rug    = params[11];
    let d_bio    = params[12]; let d_mot   = params[13]; let d_rug    = params[14];
    let stress   = params[15];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let cell = fmax_d(state[0], z); let cdg  = fmax_d(state[1], z);
    let vpsr = fmax_d(state[2], z); let bio  = fmax_d(state[3], z);
    let mot  = fmax_d(state[4], z); let rug  = fmax_d(state[5], z);

    var dy: array<f64, 6>;
    dy[0] = mu_max * cell * (one - cell / (k_cap + 1e-30)) - death * cell;
    dy[1] = stress * k_cdg * cell - d_cdg * cdg;
    let charge   = k_charge * hill_d(cdg, k_vcdg, n_vpsr) * (one - vpsr);
    let discharge = k_disch * vpsr;
    dy[2] = charge - discharge;
    dy[3] = w_bio * vpsr * (one - bio) - d_bio * bio;
    dy[4] = w_mot * (one - vpsr) * (one - mot) - d_mot * mot;
    dy[5] = w_rug * vpsr * vpsr * (one - rug) - d_rug * rug;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = CapacitorParams::from_flat(params);
        let cell = state[0].max(0.0);
        let cdg = state[1].max(0.0);
        let vpsr = state[2].max(0.0);
        let bio = state[3].max(0.0);
        let mot = state[4].max(0.0);
        let rug = state[5].max(0.0);

        let hill_v = {
            if cdg <= 0.0 {
                0.0
            } else {
                let xn = cdg.powf(p.n_vpsr);
                xn / (p.k_vpsr_cdg.powf(p.n_vpsr) + xn)
            }
        };

        let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));
        let d_cdg = (p.stress_factor * p.k_cdg_prod).mul_add(cell, -(p.d_cdg * cdg));
        let charge = p.k_vpsr_charge * hill_v * (1.0 - vpsr);
        let discharge = p.k_vpsr_discharge * vpsr;
        let d_vpsr = charge - discharge;
        let d_bio = (p.w_biofilm * vpsr).mul_add(1.0 - bio, -(p.d_bio * bio));
        let d_mot = (p.w_motility * (1.0 - vpsr)).mul_add(1.0 - mot, -(p.d_mot * mot));
        let d_rug = (p.w_rugose * vpsr * vpsr).mul_add(1.0 - rug, -(p.d_rug * rug));

        vec![d_cell, d_cdg, d_vpsr, d_bio, d_mot, d_rug]
    }
}
