// capacitor_ode_rk4_f64.wgsl — Batched RK4 for phenotypic capacitor ODE
//
// LOCAL EVOLUTION — to be absorbed by ToadStool as BatchedOdeRK4Generic<6, 16>.
//
// ODE system (Mhatre et al. 2020):
//   dN/dt = mu_max·N·(1 - N/K) - death·N
//   dC/dt = stress·k_cdg·N - d_cdg·C
//   dV/dt = k_charge·hill(C, K_cdg, n)·(1-V) - k_discharge·V
//   dB/dt = w_bio·V·(1-B) - d_bio·B
//   dM/dt = w_mot·(1-V)·(1-M) - d_mot·M
//   dR/dt = w_rug·V²·(1-R) - d_rug·R
//
// State vector: [N, C, V, B, M, R]  (6 variables)
// Parameters (per batch, 16 values — matches CapacitorParams::to_flat()):
//   [mu_max, k_cap, death_rate, k_cdg_prod, d_cdg, k_vpsr_charge,
//    k_vpsr_discharge, n_vpsr, k_vpsr_cdg, w_biofilm, w_motility,
//    w_rugose, d_bio, d_mot, d_rug, stress_factor]

// f64 is enabled by compile_shader_f64() preamble injection

const N_VARS:   u32 = 6u;
const N_PARAMS: u32 = 16u;

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
fn fpow(base: f64, exp_val: f64) -> f64 {
    let z = base - base;
    if (base <= z) { return z; }
    return exp(exp_val * log(base));
}

fn hill_n(x: f64, k: f64, n: f64) -> f64 {
    let z = x - x;
    if (x <= z) { return z; }
    let xn = fpow(x, n);
    let kn = fpow(k, n);
    return xn / (kn + xn + 1e-30);
}

fn cap_deriv(y: array<f64, 6>, p_base: u32) -> array<f64, 6> {
    let mu_max   = batch_params[p_base + 0u];
    let k_cap    = batch_params[p_base + 1u];
    let death    = batch_params[p_base + 2u];
    let k_cdg    = batch_params[p_base + 3u];
    let d_cdg    = batch_params[p_base + 4u];
    let k_charge = batch_params[p_base + 5u];
    let k_disch  = batch_params[p_base + 6u];
    let n_vpsr   = batch_params[p_base + 7u];
    let k_vcdg   = batch_params[p_base + 8u];
    let w_bio    = batch_params[p_base + 9u];
    let w_mot    = batch_params[p_base + 10u];
    let w_rug    = batch_params[p_base + 11u];
    let d_bio    = batch_params[p_base + 12u];
    let d_mot    = batch_params[p_base + 13u];
    let d_rug    = batch_params[p_base + 14u];
    let stress   = batch_params[p_base + 15u];

    let z   = y[0] - y[0];
    let one = z + 1.0;

    let cell = fmax(y[0], z);
    let cdg  = fmax(y[1], z);
    let vpsr = fmax(y[2], z);
    let bio  = fmax(y[3], z);
    let mot  = fmax(y[4], z);
    let rug  = fmax(y[5], z);

    var dy: array<f64, 6>;
    dy[0] = mu_max * cell * (one - cell / (k_cap + 1e-30)) - death * cell;
    dy[1] = stress * k_cdg * cell - d_cdg * cdg;

    let charge   = k_charge * hill_n(cdg, k_vcdg, n_vpsr) * (one - vpsr);
    let discharge = k_disch * vpsr;
    dy[2] = charge - discharge;

    dy[3] = w_bio * vpsr * (one - bio) - d_bio * bio;
    dy[4] = w_mot * (one - vpsr) * (one - mot) - d_mot * mot;
    dy[5] = w_rug * vpsr * vpsr * (one - rug) - d_rug * rug;
    return dy;
}

fn add6(a: array<f64, 6>, b: array<f64, 6>, s: f64) -> array<f64, 6> {
    var r: array<f64, 6>;
    r[0] = a[0] + s * b[0]; r[1] = a[1] + s * b[1]; r[2] = a[2] + s * b[2];
    r[3] = a[3] + s * b[3]; r[4] = a[4] + s * b[4]; r[5] = a[5] + s * b[5];
    return r;
}

fn rk4_step(y: array<f64, 6>, p_base: u32, h: f64) -> array<f64, 6> {
    let z    = h - h;
    let half = z + 0.5;
    let two  = z + 2.0;
    let h2   = h * half;

    let k1 = cap_deriv(y, p_base);
    let k2 = cap_deriv(add6(y, k1, h2), p_base);
    let k3 = cap_deriv(add6(y, k2, h2), p_base);
    let k4 = cap_deriv(add6(y, k3, h), p_base);

    let sixth = (z + 1.0) / (z + 6.0);
    var yn: array<f64, 6>;
    yn[0] = y[0] + h * sixth * (k1[0] + two * k2[0] + two * k3[0] + k4[0]);
    yn[1] = y[1] + h * sixth * (k1[1] + two * k2[1] + two * k3[1] + k4[1]);
    yn[2] = y[2] + h * sixth * (k1[2] + two * k2[2] + two * k3[2] + k4[2]);
    yn[3] = y[3] + h * sixth * (k1[3] + two * k2[3] + two * k3[3] + k4[3]);
    yn[4] = y[4] + h * sixth * (k1[4] + two * k2[4] + two * k3[4] + k4[4]);
    yn[5] = y[5] + h * sixth * (k1[5] + two * k2[5] + two * k3[5] + k4[5]);
    return yn;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    if (b >= config.n_batches) { return; }

    let s_base = b * N_VARS;
    let p_base = b * N_PARAMS;

    var y: array<f64, 6>;
    y[0] = initial_states[s_base + 0u];
    y[1] = initial_states[s_base + 1u];
    y[2] = initial_states[s_base + 2u];
    y[3] = initial_states[s_base + 3u];
    y[4] = initial_states[s_base + 4u];
    y[5] = initial_states[s_base + 5u];

    let cmax = config.clamp_max;
    let cmin = config.clamp_min;
    let h    = config.h;

    for (var step = 0u; step < config.n_steps; step = step + 1u) {
        let yn = rk4_step(y, p_base, h);
        y[0] = fclamp(yn[0], cmin, cmax);
        y[1] = fclamp(yn[1], cmin, cmax);
        y[2] = fclamp(yn[2], cmin, cmin + 1.0);
        y[3] = fclamp(yn[3], cmin, cmin + 1.0);
        y[4] = fclamp(yn[4], cmin, cmin + 1.0);
        y[5] = fclamp(yn[5], cmin, cmin + 1.0);
    }

    output_states[s_base + 0u] = y[0];
    output_states[s_base + 1u] = y[1];
    output_states[s_base + 2u] = y[2];
    output_states[s_base + 3u] = y[3];
    output_states[s_base + 4u] = y[4];
    output_states[s_base + 5u] = y[5];
}
