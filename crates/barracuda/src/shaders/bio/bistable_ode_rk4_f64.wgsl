// SPDX-License-Identifier: AGPL-3.0-only
// bistable_ode_rk4_f64.wgsl — Batched RK4 for bistable QS/biofilm ODE
//
// LOCAL EVOLUTION — to be absorbed by ToadStool.
//
// Extends the Waters 2008 QS/c-di-GMP model with cooperative feedback:
//   DGC_rate += alpha_fb * Hill(B, k_fb, n_fb)
// This creates a bistable switch with hysteresis in the biofilm commitment.
//
// State vector: [N, AI, HapR, c-di-GMP, Biofilm]  (5 variables)
// Parameters (per batch, 21 — matches BistableParams::to_flat()):
//   [mu_max, k_cap, death_rate, k_ai_prod, d_ai,
//    k_hapr_max, k_hapr_ai, n_hapr, d_hapr,
//    k_dgc_basal, k_dgc_rep, k_pde_basal, k_pde_act, d_cdg,
//    k_bio_max, k_bio_cdg, n_bio, d_bio,
//    alpha_fb, n_fb, k_fb]

const N_VARS:   u32 = 5u;
const N_PARAMS: u32 = 21u;

struct OdeConfig {
    n_batches:      u32,
    n_steps:        u32,
    _pad0:          u32,
    _pad1:          u32,
    h:              f64,
    t0:             f64,
    clamp_max:      f64,
    clamp_min:      f64,
}

@group(0) @binding(0) var<uniform>             config:         OdeConfig;
@group(0) @binding(1) var<storage, read>       initial_states: array<f64>;
@group(0) @binding(2) var<storage, read>       batch_params:   array<f64>;
@group(0) @binding(3) var<storage, read_write> output_states:  array<f64>;

fn fmax(a: f64, b: f64) -> f64 {
    if (a >= b) { return a; }
    return b;
}
fn fclamp(x: f64, lo: f64, hi: f64) -> f64 {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}
fn fpow(base: f64, e: f64) -> f64 {
    return exp_f64(e * log_f64(base));
}

fn hill(x: f64, K: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax(x, z);
    let xn = fpow(xc, n);
    let Kn = fpow(fmax(K, z + 1e-30), n);
    return xn / (Kn + xn);
}

fn bistable_deriv(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64,
                  p_base: u32) -> array<f64, 5> {
    let mu_max     = batch_params[p_base + 0u];
    let k_cap      = batch_params[p_base + 1u];
    let death_rate = batch_params[p_base + 2u];
    let k_ai_prod  = batch_params[p_base + 3u];
    let d_ai       = batch_params[p_base + 4u];
    let k_hapr_max = batch_params[p_base + 5u];
    let k_hapr_ai  = batch_params[p_base + 6u];
    let n_hapr     = batch_params[p_base + 7u];
    let d_hapr     = batch_params[p_base + 8u];
    let k_dgc_bas  = batch_params[p_base + 9u];
    let k_dgc_rep  = batch_params[p_base + 10u];
    let k_pde_bas  = batch_params[p_base + 11u];
    let k_pde_act  = batch_params[p_base + 12u];
    let d_cdg      = batch_params[p_base + 13u];
    let k_bio_max  = batch_params[p_base + 14u];
    let k_bio_cdg  = batch_params[p_base + 15u];
    let n_bio      = batch_params[p_base + 16u];
    let d_bio      = batch_params[p_base + 17u];
    let alpha_fb   = batch_params[p_base + 18u];
    let n_fb       = batch_params[p_base + 19u];
    let k_fb       = batch_params[p_base + 20u];

    let z = y0 - y0;
    let one = z + 1.0;

    let cell = fmax(y0, z);
    let ai   = fmax(y1, z);
    let hapr = fmax(y2, z);
    let cdg  = fmax(y3, z);
    let bio  = fmax(y4, z);

    var dy: array<f64, 5>;

    // dN/dt — logistic growth - death
    dy[0] = mu_max * cell * (one - cell / fmax(k_cap, z + 1e-30)) - death_rate * cell;
    // dAI/dt
    dy[1] = k_ai_prod * cell - d_ai * ai;
    // dHapR/dt
    dy[2] = k_hapr_max * hill(ai, k_hapr_ai, n_hapr) - d_hapr * hapr;
    // dC/dt — basal DGC + feedback DGC - PDE - dilution
    let basal_dgc = k_dgc_bas * fmax(one - k_dgc_rep * hapr, z);
    let feedback_dgc = alpha_fb * hill(bio, k_fb, n_fb);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[3] = basal_dgc + feedback_dgc - pde_rate * cdg - d_cdg * cdg;
    // dB/dt — biofilm promotion - dispersal
    dy[4] = k_bio_max * hill(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;

    return dy;
}

fn rk4_step(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64,
            p_base: u32, h: f64) -> array<f64, 5> {
    let z = h - h;
    let half = z + 0.5;
    let two  = z + 2.0;
    let h2 = h * half;

    let k1 = bistable_deriv(y0, y1, y2, y3, y4, p_base);
    let k2 = bistable_deriv(
        y0 + h2 * k1[0], y1 + h2 * k1[1], y2 + h2 * k1[2],
        y3 + h2 * k1[3], y4 + h2 * k1[4], p_base);
    let k3 = bistable_deriv(
        y0 + h2 * k2[0], y1 + h2 * k2[1], y2 + h2 * k2[2],
        y3 + h2 * k2[3], y4 + h2 * k2[4], p_base);
    let k4 = bistable_deriv(
        y0 + h * k3[0], y1 + h * k3[1], y2 + h * k3[2],
        y3 + h * k3[3], y4 + h * k3[4], p_base);

    var yn: array<f64, 5>;
    let sixth = (z + 1.0) / (z + 6.0);
    yn[0] = y0 + h * sixth * (k1[0] + two * k2[0] + two * k3[0] + k4[0]);
    yn[1] = y1 + h * sixth * (k1[1] + two * k2[1] + two * k3[1] + k4[1]);
    yn[2] = y2 + h * sixth * (k1[2] + two * k2[2] + two * k3[2] + k4[2]);
    yn[3] = y3 + h * sixth * (k1[3] + two * k2[3] + two * k3[3] + k4[3]);
    yn[4] = y4 + h * sixth * (k1[4] + two * k2[4] + two * k3[4] + k4[4]);
    return yn;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= config.n_batches) { return; }

    let s_base = b * N_VARS;
    let p_base = b * N_PARAMS;

    var y0 = initial_states[s_base + 0u];
    var y1 = initial_states[s_base + 1u];
    var y2 = initial_states[s_base + 2u];
    var y3 = initial_states[s_base + 3u];
    var y4 = initial_states[s_base + 4u];

    let cmax = config.clamp_max;
    let cmin = config.clamp_min;
    let h    = config.h;

    for (var step = 0u; step < config.n_steps; step = step + 1u) {
        let yn = rk4_step(y0, y1, y2, y3, y4, p_base, h);
        y0 = fclamp(yn[0], cmin, cmax);
        y1 = fclamp(yn[1], cmin, cmax);
        y2 = fclamp(yn[2], cmin, cmax);
        y3 = fclamp(yn[3], cmin, cmax);
        let one = cmin - cmin + 1.0;
        y4 = fclamp(yn[4], cmin, one);
    }

    output_states[s_base + 0u] = y0;
    output_states[s_base + 1u] = y1;
    output_states[s_base + 2u] = y2;
    output_states[s_base + 3u] = y3;
    output_states[s_base + 4u] = y4;
}
