// SPDX-License-Identifier: AGPL-3.0-or-later
// cooperation_ode_rk4_f64.wgsl — Batched RK4 for cooperative QS game theory
//
// LOCAL EVOLUTION — to be absorbed by ToadStool as BatchedOdeRK4Generic<4, 13>.
//
// ODE system (Bruger & Waters 2018):
//   dNc/dt = (mu_c - cost + benefit·hill(A) + disp·(1-B)) · crowding · Nc - death·Nc
//   dNd/dt = (mu_d + benefit·hill(A) + disp·(1-B)) · crowding · Nd - death·Nd
//   dA/dt  = k_ai·Nc - d_ai·A
//   dB/dt  = k_bio·hill(A, k_bio_ai)·(1-B) - d_bio·B
//
// State vector: [Nc, Nd, A, B]  (4 variables)
// Parameters (per batch, 13 values — matches CooperationParams::to_flat()):
//   [mu_coop, mu_cheat, k_cap, death_rate, k_ai_prod, d_ai,
//    benefit, k_benefit, cost, k_bio, k_bio_ai, dispersal_bonus, d_bio]

// f64 is enabled by compile_shader_f64() preamble injection

const N_VARS:   u32 = 4u;
const N_PARAMS: u32 = 13u;

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

fn hill(x: f64, k: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let x2 = x * x;
    return x2 / (k * k + x2);
}

fn coop_deriv(y0: f64, y1: f64, y2: f64, y3: f64,
              p_base: u32) -> array<f64, 4> {
    let mu_coop  = batch_params[p_base + 0u];
    let mu_cheat = batch_params[p_base + 1u];
    let k_cap    = batch_params[p_base + 2u];
    let death    = batch_params[p_base + 3u];
    let k_ai     = batch_params[p_base + 4u];
    let d_ai     = batch_params[p_base + 5u];
    let benefit  = batch_params[p_base + 6u];
    let k_ben    = batch_params[p_base + 7u];
    let cost     = batch_params[p_base + 8u];
    let k_bio    = batch_params[p_base + 9u];
    let k_bio_ai = batch_params[p_base + 10u];
    let disp     = batch_params[p_base + 11u];
    let d_bio    = batch_params[p_base + 12u];

    let z   = y0 - y0;
    let one = z + 1.0;

    let nc  = fmax(y0, z);
    let nd  = fmax(y1, z);
    let ai  = fmax(y2, z);
    let bio = fmax(y3, z);

    let n_total  = nc + nd;
    let crowding = fmax(one - n_total / (k_cap + 1e-30), z);

    let sig_benefit = benefit * hill(ai, k_ben);
    let dispersal   = disp * (one - bio);

    let fit_coop  = (mu_coop  - cost + sig_benefit + dispersal) * crowding;
    let fit_cheat = (mu_cheat + sig_benefit + dispersal) * crowding;

    var dy: array<f64, 4>;
    dy[0] = fit_coop  * nc - death * nc;
    dy[1] = fit_cheat * nd - death * nd;
    dy[2] = k_ai * nc - d_ai * ai;
    dy[3] = k_bio * hill(ai, k_bio_ai) * (one - bio) - d_bio * bio;
    return dy;
}

fn rk4_step(y0: f64, y1: f64, y2: f64, y3: f64,
            p_base: u32, h: f64) -> array<f64, 4> {
    let z    = h - h;
    let half = z + 0.5;
    let two  = z + 2.0;
    let h2   = h * half;

    let k1 = coop_deriv(y0, y1, y2, y3, p_base);
    let k2 = coop_deriv(
        y0 + h2 * k1[0], y1 + h2 * k1[1],
        y2 + h2 * k1[2], y3 + h2 * k1[3], p_base);
    let k3 = coop_deriv(
        y0 + h2 * k2[0], y1 + h2 * k2[1],
        y2 + h2 * k2[2], y3 + h2 * k2[3], p_base);
    let k4 = coop_deriv(
        y0 + h * k3[0], y1 + h * k3[1],
        y2 + h * k3[2], y3 + h * k3[3], p_base);

    var yn: array<f64, 4>;
    let sixth = (z + 1.0) / (z + 6.0);
    yn[0] = y0 + h * sixth * (k1[0] + two * k2[0] + two * k3[0] + k4[0]);
    yn[1] = y1 + h * sixth * (k1[1] + two * k2[1] + two * k3[1] + k4[1]);
    yn[2] = y2 + h * sixth * (k1[2] + two * k2[2] + two * k3[2] + k4[2]);
    yn[3] = y3 + h * sixth * (k1[3] + two * k2[3] + two * k3[3] + k4[3]);
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

    let cmax = config.clamp_max;
    let cmin = config.clamp_min;
    let h    = config.h;

    for (var step = 0u; step < config.n_steps; step = step + 1u) {
        let yn = rk4_step(y0, y1, y2, y3, p_base, h);
        y0 = fclamp(yn[0], cmin, cmax);
        y1 = fclamp(yn[1], cmin, cmax);
        y2 = fclamp(yn[2], cmin, cmax);
        // biofilm clamped to [0, 1]
        y3 = fclamp(yn[3], cmin, cmin + 1.0);
    }

    output_states[s_base + 0u] = y0;
    output_states[s_base + 1u] = y1;
    output_states[s_base + 2u] = y2;
    output_states[s_base + 3u] = y3;
}
