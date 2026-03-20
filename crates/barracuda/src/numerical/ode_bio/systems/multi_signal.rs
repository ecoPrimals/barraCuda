// SPDX-License-Identifier: AGPL-3.0-or-later

#![expect(clippy::needless_raw_string_hashes, reason = "WGSL readability")]

use super::super::params::{MULTI_SIGNAL_N_PARAMS, MULTI_SIGNAL_N_VARS, MultiSignalParams};
use crate::numerical::ode_generic::OdeSystem;

/// Dual-signal QS regulatory network for `BatchedOdeRK4<MultiSignalOde>`.
pub struct MultiSignalOde;

impl OdeSystem for MultiSignalOde {
    const N_VARS: usize = MULTI_SIGNAL_N_VARS;
    const N_PARAMS: usize = MULTI_SIGNAL_N_PARAMS;

    fn system_name() -> &'static str {
        "multi_signal"
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
fn hill_repress_d(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax_d(x, z);
    let kn = fpow_d(fmax_d(k, z + 1e-30), n);
    let xn = fpow_d(xc, n);
    return kn / (kn + xn);
}
fn deriv(state: array<f64, 7>, params: array<f64, 24>, t: f64) -> array<f64, 7> {
    let mu_max     = params[0];  let k_cap      = params[1];  let death_rate = params[2];
    let k_cai1     = params[3];  let d_cai1     = params[4];  let k_cqs      = params[5];
    let k_ai2      = params[6];  let d_ai2      = params[7];  let k_luxpq    = params[8];
    let k_luxo_p   = params[9];  let d_luxo_p   = params[10]; let k_hapr_max = params[11];
    let n_repress  = params[12]; let k_repress  = params[13]; let d_hapr     = params[14];
    let k_dgc_bas  = params[15]; let k_dgc_rep  = params[16]; let k_pde_bas  = params[17];
    let k_pde_act  = params[18]; let d_cdg      = params[19]; let k_bio_max  = params[20];
    let k_bio_cdg  = params[21]; let n_bio      = params[22]; let d_bio      = params[23];

    let z   = state[0] - state[0];
    let one = z + 1.0;
    let two = z + 2.0;
    let cell   = fmax_d(state[0], z); let cai1   = fmax_d(state[1], z);
    let ai2    = fmax_d(state[2], z); let luxo_p = fmax_d(state[3], z);
    let hapr   = fmax_d(state[4], z); let cdg    = fmax_d(state[5], z);
    let bio    = fmax_d(state[6], z);

    var dy: array<f64, 7>;
    dy[0] = mu_max * cell * (one - cell / fmax_d(k_cap, z + 1e-30)) - death_rate * cell;
    dy[1] = k_cai1 * cell - d_cai1 * cai1;
    dy[2] = k_ai2 * cell - d_ai2 * ai2;
    let dephos_cai1 = hill_d(cai1, k_cqs, two);
    let dephos_ai2 = hill_d(ai2, k_luxpq, two);
    dy[3] = k_luxo_p - (d_luxo_p + dephos_cai1 + dephos_ai2) * luxo_p;
    dy[4] = k_hapr_max * hill_repress_d(luxo_p, k_repress, n_repress) - d_hapr * hapr;
    let dgc_rate = k_dgc_bas * fmax_d(one - k_dgc_rep * hapr, z);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[5] = dgc_rate - pde_rate * cdg - d_cdg * cdg;
    dy[6] = k_bio_max * hill_d(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;
    return dy;
}
"#
    }

    fn cpu_derivative(_t: f64, state: &[f64], params: &[f64]) -> Vec<f64> {
        let p = MultiSignalParams::from_flat(params);
        let cell = state[0].max(0.0);
        let cai1 = state[1].max(0.0);
        let ai2 = state[2].max(0.0);
        let luxo_p = state[3].max(0.0);
        let hapr = state[4].max(0.0);
        let cdg = state[5].max(0.0);
        let bio = state[6].max(0.0);

        let hill = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            let xn = x.powf(n);
            xn / (k.powf(n) + xn)
        };
        let hill_repress = |x: f64, k: f64, n: f64| -> f64 {
            if x <= 0.0 {
                return 1.0;
            }
            let kn = k.powf(n);
            kn / (kn + x.powf(n))
        };

        let d_cell = (p.mu_max * cell).mul_add(1.0 - cell / p.k_cap, -(p.death_rate * cell));
        let d_cai1 = p.k_cai1_prod.mul_add(cell, -p.d_cai1 * cai1);
        let d_ai2 = p.k_ai2_prod.mul_add(cell, -p.d_ai2 * ai2);

        let dephos_cai1 = hill(cai1, p.k_cqs, 2.0);
        let dephos_ai2 = hill(ai2, p.k_luxpq, 2.0);
        let d_luxo_p = (p.d_luxo_p + dephos_cai1 + dephos_ai2).mul_add(-luxo_p, p.k_luxo_phos);

        let d_hapr = p.k_hapr_max.mul_add(
            hill_repress(luxo_p, p.k_repress, p.n_repress),
            -p.d_hapr * hapr,
        );

        let dgc_rate = p.k_dgc_basal * p.k_dgc_rep.mul_add(-hapr, 1.0).max(0.0);
        let pde_rate = p.k_pde_act.mul_add(hapr, p.k_pde_basal);
        let d_cdg = p.d_cdg.mul_add(-cdg, dgc_rate - pde_rate * cdg);

        let bio_promote = p.k_bio_max * hill(cdg, p.k_bio_cdg, p.n_bio);
        let d_bio = bio_promote.mul_add(1.0 - bio, -(p.d_bio * bio));

        vec![d_cell, d_cai1, d_ai2, d_luxo_p, d_hapr, d_cdg, d_bio]
    }
}
