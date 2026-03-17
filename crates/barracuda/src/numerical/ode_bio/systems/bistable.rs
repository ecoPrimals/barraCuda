// SPDX-License-Identifier: AGPL-3.0-only

#![expect(clippy::needless_raw_string_hashes, reason = "WGSL readability")]

use super::super::params::{BISTABLE_N_PARAMS, BISTABLE_N_VARS, BistableParams};
use crate::numerical::ode_generic::OdeSystem;

/// Bistable phenotypic switching ODE for `BatchedOdeRK4<BistableOde>`.
pub struct BistableOde;

impl OdeSystem for BistableOde {
    const N_VARS: usize = BISTABLE_N_VARS;
    const N_PARAMS: usize = BISTABLE_N_PARAMS;

    fn system_name() -> &'static str {
        "bistable"
    }

    fn wgsl_derivative() -> &'static str {
        r#"
fn fmax_d(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fpow_d(base: f64, e: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(e * log(base));
}
fn hill_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax_d(x, z);
    let xn = fpow_d(xc, n);
    let kn = fpow_d(fmax_d(k, z + 1e-30), n);
    return xn / (kn + xn);
}
fn deriv(state: array<f64, 5>, params: array<f64, 21>, t: f64) -> array<f64, 5> {
    let mu_max     = params[0];  let k_cap      = params[1];  let death_rate = params[2];
    let k_ai_prod  = params[3];  let d_ai       = params[4];  let k_hapr_max = params[5];
    let k_hapr_ai  = params[6];  let n_hapr     = params[7];  let d_hapr     = params[8];
    let k_dgc_bas  = params[9];  let k_dgc_rep  = params[10]; let k_pde_bas  = params[11];
    let k_pde_act  = params[12]; let d_cdg      = params[13]; let k_bio_max  = params[14];
    let k_bio_cdg  = params[15]; let n_bio      = params[16]; let d_bio      = params[17];
    let alpha_fb   = params[18]; let n_fb       = params[19]; let k_fb       = params[20];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let cell = fmax_d(state[0], z); let ai   = fmax_d(state[1], z);
    let hapr = fmax_d(state[2], z); let cdg  = fmax_d(state[3], z);
    let bio  = fmax_d(state[4], z);

    var dy: array<f64, 5>;
    dy[0] = mu_max * cell * (one - cell / fmax_d(k_cap, z + 1e-30)) - death_rate * cell;
    dy[1] = k_ai_prod * cell - d_ai * ai;
    dy[2] = k_hapr_max * hill_d(ai, k_hapr_ai, n_hapr) - d_hapr * hapr;
    let basal_dgc = k_dgc_bas * fmax_d(one - k_dgc_rep * hapr, z);
    let feedback_dgc = alpha_fb * hill_d(bio, k_fb, n_fb);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[3] = basal_dgc + feedback_dgc - pde_rate * cdg - d_cdg * cdg;
    dy[4] = k_bio_max * hill_d(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = BistableParams::from_flat(params);
        let cell = state[0].max(0.0);
        let ai = state[1].max(0.0);
        let hapr = state[2].max(0.0);
        let cdg = state[3].max(0.0);
        let bio = state[4].max(0.0);

        let b = &p.base;
        let hill = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let xn = x.powf(n);
            xn / (k.powf(n) + xn)
        };

        let d_cell = (b.mu_max * cell).mul_add(1.0 - cell / b.k_cap, -(b.death_rate * cell));
        let d_ai = b.k_ai_prod.mul_add(cell, -b.d_ai * ai);
        let d_hapr = b
            .k_hapr_max
            .mul_add(hill(ai, b.k_hapr_ai, b.n_hapr), -b.d_hapr * hapr);

        let basal_dgc = b.k_dgc_basal * b.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
        let feedback_dgc = p.alpha_fb * hill(bio, p.k_fb, p.n_fb);
        let pde_rate = b.k_pde_act.mul_add(hapr, b.k_pde_basal);
        let d_cdg = b
            .d_cdg
            .mul_add(-cdg, basal_dgc + feedback_dgc - pde_rate * cdg);

        let bio_promote = b.k_bio_max * hill(cdg, b.k_bio_cdg, b.n_bio);
        let d_bio = bio_promote.mul_add(1.0 - bio, -(b.d_bio * bio));

        vec![d_cell, d_ai, d_hapr, d_cdg, d_bio]
    }
}
