// multi_signal_ode_rk4_f64.wgsl — Batched RK4 for dual-signal QS ODE
//
// LOCAL EVOLUTION — to be absorbed by ToadStool.
//
// V. cholerae dual-signal quorum sensing: CAI-1 and AI-2 independently
// drive LuxO dephosphorylation → HapR activation → c-di-GMP → biofilm.
//
// State vector: [N, CAI1, AI2, LuxO~P, HapR, c-di-GMP, Biofilm]  (7 variables)
// Parameters (per batch, 24 — matches MultiSignalParams::to_flat()):
//   [mu_max, k_cap, death_rate, k_cai1_prod, d_cai1, k_cqs,
//    k_ai2_prod, d_ai2, k_luxpq, k_luxo_phos, d_luxo_p,
//    k_hapr_max, n_repress, k_repress, d_hapr,
//    k_dgc_basal, k_dgc_rep, k_pde_basal, k_pde_act, d_cdg,
//    k_bio_max, k_bio_cdg, n_bio, d_bio]

const N_VARS:   u32 = 7u;
const N_PARAMS: u32 = 24u;

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

fn hill_repress(x: f64, K: f64, n: f64) -> f64 {
    let z = x - x;
    let xc = fmax(x, z);
    let Kn = fpow(fmax(K, z + 1e-30), n);
    let xn = fpow(xc, n);
    return Kn / (Kn + xn);
}

fn ms_deriv(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64, y5: f64, y6: f64,
            p_base: u32) -> array<f64, 7> {
    let mu_max     = batch_params[p_base + 0u];
    let k_cap      = batch_params[p_base + 1u];
    let death_rate = batch_params[p_base + 2u];
    let k_cai1     = batch_params[p_base + 3u];
    let d_cai1     = batch_params[p_base + 4u];
    let k_cqs      = batch_params[p_base + 5u];
    let k_ai2      = batch_params[p_base + 6u];
    let d_ai2      = batch_params[p_base + 7u];
    let k_luxpq    = batch_params[p_base + 8u];
    let k_luxo_p   = batch_params[p_base + 9u];
    let d_luxo_p   = batch_params[p_base + 10u];
    let k_hapr_max = batch_params[p_base + 11u];
    let n_repress  = batch_params[p_base + 12u];
    let k_repress  = batch_params[p_base + 13u];
    let d_hapr     = batch_params[p_base + 14u];
    let k_dgc_bas  = batch_params[p_base + 15u];
    let k_dgc_rep  = batch_params[p_base + 16u];
    let k_pde_bas  = batch_params[p_base + 17u];
    let k_pde_act  = batch_params[p_base + 18u];
    let d_cdg      = batch_params[p_base + 19u];
    let k_bio_max  = batch_params[p_base + 20u];
    let k_bio_cdg  = batch_params[p_base + 21u];
    let n_bio      = batch_params[p_base + 22u];
    let d_bio      = batch_params[p_base + 23u];

    let z = y0 - y0;
    let one = z + 1.0;
    let two = z + 2.0;

    let cell   = fmax(y0, z);
    let cai1   = fmax(y1, z);
    let ai2    = fmax(y2, z);
    let luxo_p = fmax(y3, z);
    let hapr   = fmax(y4, z);
    let cdg    = fmax(y5, z);
    let bio    = fmax(y6, z);

    var dy: array<f64, 7>;

    // dN/dt — logistic growth - death
    dy[0] = mu_max * cell * (one - cell / fmax(k_cap, z + 1e-30)) - death_rate * cell;
    // dCAI1/dt
    dy[1] = k_cai1 * cell - d_cai1 * cai1;
    // dAI2/dt
    dy[2] = k_ai2 * cell - d_ai2 * ai2;
    // dLuxO~P/dt — phosphorylation, dephosphorylated by both signals
    let dephos_cai1 = hill(cai1, k_cqs, two);
    let dephos_ai2 = hill(ai2, k_luxpq, two);
    let total_dephos = dephos_cai1 + dephos_ai2;
    dy[3] = k_luxo_p - (d_luxo_p + total_dephos) * luxo_p;
    // dHapR/dt — repressed by LuxO~P
    dy[4] = k_hapr_max * hill_repress(luxo_p, k_repress, n_repress) - d_hapr * hapr;
    // dc-di-GMP/dt
    let dgc_rate = k_dgc_bas * fmax(one - k_dgc_rep * hapr, z);
    let pde_rate = k_pde_bas + k_pde_act * hapr;
    dy[5] = dgc_rate - pde_rate * cdg - d_cdg * cdg;
    // dBiofilm/dt
    dy[6] = k_bio_max * hill(cdg, k_bio_cdg, n_bio) * (one - bio) - d_bio * bio;

    return dy;
}

fn rk4_step(y0: f64, y1: f64, y2: f64, y3: f64, y4: f64, y5: f64, y6: f64,
            p_base: u32, h: f64) -> array<f64, 7> {
    let z = h - h;
    let half = z + 0.5;
    let two  = z + 2.0;
    let h2 = h * half;

    let k1 = ms_deriv(y0, y1, y2, y3, y4, y5, y6, p_base);
    let k2 = ms_deriv(
        y0 + h2 * k1[0], y1 + h2 * k1[1], y2 + h2 * k1[2],
        y3 + h2 * k1[3], y4 + h2 * k1[4], y5 + h2 * k1[5],
        y6 + h2 * k1[6], p_base);
    let k3 = ms_deriv(
        y0 + h2 * k2[0], y1 + h2 * k2[1], y2 + h2 * k2[2],
        y3 + h2 * k2[3], y4 + h2 * k2[4], y5 + h2 * k2[5],
        y6 + h2 * k2[6], p_base);
    let k4 = ms_deriv(
        y0 + h * k3[0], y1 + h * k3[1], y2 + h * k3[2],
        y3 + h * k3[3], y4 + h * k3[4], y5 + h * k3[5],
        y6 + h * k3[6], p_base);

    var yn: array<f64, 7>;
    let sixth = (z + 1.0) / (z + 6.0);
    yn[0] = y0 + h * sixth * (k1[0] + two * k2[0] + two * k3[0] + k4[0]);
    yn[1] = y1 + h * sixth * (k1[1] + two * k2[1] + two * k3[1] + k4[1]);
    yn[2] = y2 + h * sixth * (k1[2] + two * k2[2] + two * k3[2] + k4[2]);
    yn[3] = y3 + h * sixth * (k1[3] + two * k2[3] + two * k3[3] + k4[3]);
    yn[4] = y4 + h * sixth * (k1[4] + two * k2[4] + two * k3[4] + k4[4]);
    yn[5] = y5 + h * sixth * (k1[5] + two * k2[5] + two * k3[5] + k4[5]);
    yn[6] = y6 + h * sixth * (k1[6] + two * k2[6] + two * k3[6] + k4[6]);
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
    var y5 = initial_states[s_base + 5u];
    var y6 = initial_states[s_base + 6u];

    let cmax = config.clamp_max;
    let cmin = config.clamp_min;
    let h    = config.h;

    for (var step = 0u; step < config.n_steps; step = step + 1u) {
        let yn = rk4_step(y0, y1, y2, y3, y4, y5, y6, p_base, h);
        y0 = fclamp(yn[0], cmin, cmax);
        y1 = fclamp(yn[1], cmin, cmax);
        y2 = fclamp(yn[2], cmin, cmax);
        y3 = fclamp(yn[3], cmin, cmax);
        y4 = fclamp(yn[4], cmin, cmax);
        y5 = fclamp(yn[5], cmin, cmax);
        let one = cmin - cmin + 1.0;
        y6 = fclamp(yn[6], cmin, one);
    }

    output_states[s_base + 0u] = y0;
    output_states[s_base + 1u] = y1;
    output_states[s_base + 2u] = y2;
    output_states[s_base + 3u] = y3;
    output_states[s_base + 4u] = y4;
    output_states[s_base + 5u] = y5;
    output_states[s_base + 6u] = y6;
}
