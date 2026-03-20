// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::needless_raw_string_hashes, reason = "WGSL readability")]

use super::super::params::{COOPERATION_N_PARAMS, COOPERATION_N_VARS, CooperationParams};
use crate::numerical::ode_generic::OdeSystem;

/// Cooperative QS game-theory ODE for `BatchedOdeRK4<CooperationOde>`.
pub struct CooperationOde;

impl OdeSystem for CooperationOde {
    const N_VARS: usize = COOPERATION_N_VARS;
    const N_PARAMS: usize = COOPERATION_N_PARAMS;

    fn system_name() -> &'static str {
        "cooperation"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn hill2_d(x: f64, k: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let x2 = x * x;
    return x2 / (k * k + x2);
}
fn deriv(state: array<f64, 4>, params: array<f64, 13>, t: f64) -> array<f64, 4> {
    let mu_coop  = params[0];  let mu_cheat = params[1];  let k_cap    = params[2];
    let death    = params[3];  let k_ai     = params[4];  let d_ai     = params[5];
    let benefit  = params[6];  let k_ben    = params[7];  let cost     = params[8];
    let k_bio    = params[9];  let k_bio_ai = params[10]; let disp     = params[11];
    let d_bio    = params[12];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let nc  = fmax_d(state[0], z); let nd  = fmax_d(state[1], z);
    let ai  = fmax_d(state[2], z); let bio = fmax_d(state[3], z);

    let n_total  = nc + nd;
    let crowding = fmax_d(one - n_total / (k_cap + 1e-30), z);
    let sig_benefit = benefit * hill2_d(ai, k_ben);
    let dispersal   = disp * (one - bio);
    let fit_coop  = (mu_coop  - cost + sig_benefit + dispersal) * crowding;
    let fit_cheat = (mu_cheat + sig_benefit + dispersal) * crowding;

    var dy: array<f64, 4>;
    dy[0] = fit_coop  * nc - death * nc;
    dy[1] = fit_cheat * nd - death * nd;
    dy[2] = k_ai * nc - d_ai * ai;
    dy[3] = k_bio * hill2_d(ai, k_bio_ai) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = CooperationParams::from_flat(params);
        let nc = state[0].max(0.0);
        let nd = state[1].max(0.0);
        let ai = state[2].max(0.0);
        let bio = state[3].max(0.0);

        let hill2 = |x: f64, k: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let x2 = x * x;
            x2 / k.mul_add(k, x2)
        };

        let n_total = nc + nd;
        let crowding = (1.0 - n_total / p.k_cap).max(0.0);
        let signal_benefit = p.benefit * hill2(ai, p.k_benefit);
        let dispersal = p.dispersal_bonus * (1.0 - bio);

        let fitness_coop = (p.mu_coop - p.cost + signal_benefit + dispersal) * crowding;
        let fitness_cheat = (p.mu_cheat + signal_benefit + dispersal) * crowding;

        vec![
            fitness_coop.mul_add(nc, -(p.death_rate * nc)),
            fitness_cheat.mul_add(nd, -(p.death_rate * nd)),
            p.k_ai_prod.mul_add(nc, -p.d_ai * ai),
            (p.k_bio * hill2(ai, p.k_bio_ai)).mul_add(1.0 - bio, -(p.d_bio * bio)),
        ]
    }
}
