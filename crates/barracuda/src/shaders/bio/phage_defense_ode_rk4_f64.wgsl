// phage_defense_ode_rk4_f64.wgsl — Batched RK4 for phage-bacteria defense ODE
//
// LOCAL EVOLUTION — to be absorbed by ToadStool.
//
// ODE system:
//   dBd/dt = (mu_max * (1 - cost) * monod(R, K_r) * Bd) - (death * Bd)
//            - (ads * Bd * P * (1 - eff))
//   dBu/dt = (mu_max * monod(R, K_r) * Bu) - (death * Bu) - (ads * Bu * P)
//   dP/dt  = burst * (ads*Bu*P + (1-eff)*ads*Bd*P) - ads*(Bd+Bu)*P - decay*P
//   dR/dt  = inflow - yield * (mu_d*Bd + mu_u*Bu) - dilution*R
//
// State vector: [Bd, Bu, P, R]  (4 variables)
// Parameters (per batch, 11 values — matches PhageDefenseParams::to_flat()):
//   [mu_max, defense_cost, k_resource, yield_coeff, adsorption_rate,
//    burst_size, defense_efficiency, phage_decay, resource_inflow,
//    resource_dilution, death_rate]

// f64 is enabled by compile_shader_f64() preamble injection

const N_VARS:   u32 = 4u;
const N_PARAMS: u32 = 11u;

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

fn monod(r: f64, k: f64) -> f64 {
    let z = r - r;
    return r / (k + r + z + 1e-30);
}

fn defense_deriv(y0: f64, y1: f64, y2: f64, y3: f64,
                 p_base: u32) -> array<f64, 4> {
    // Must match PhageDefenseParams::to_flat() ordering
    let mu_max = batch_params[p_base + 0u];
    let cost   = batch_params[p_base + 1u];  // defense_cost
    let k_res  = batch_params[p_base + 2u];  // k_resource
    let yld    = batch_params[p_base + 3u];  // yield_coeff
    let ads    = batch_params[p_base + 4u];  // adsorption_rate
    let burst  = batch_params[p_base + 5u];  // burst_size
    let eff    = batch_params[p_base + 6u];  // defense_efficiency
    let decay  = batch_params[p_base + 7u];  // phage_decay
    let inflow = batch_params[p_base + 8u];  // resource_inflow
    let dilut  = batch_params[p_base + 9u];  // resource_dilution
    let death  = batch_params[p_base + 10u]; // death_rate

    let z = y0 - y0; // f64 zero
    let one = z + 1.0;

    let bd    = fmax(y0, z);
    let bu    = fmax(y1, z);
    let phage = fmax(y2, z);
    let r     = fmax(y3, z);

    let growth_limit = monod(r, k_res);
    let mu_d = mu_max * (one - cost) * growth_limit;
    let mu_u = mu_max * growth_limit;

    let inf_d = ads * bd * phage;
    let inf_u = ads * bu * phage;
    let kill_d = inf_d * (one - eff);

    var dy: array<f64, 4>;
    // dBd/dt
    dy[0] = mu_d * bd - death * bd - kill_d;
    // dBu/dt
    dy[1] = mu_u * bu - death * bu - inf_u;
    // dP/dt
    dy[2] = burst * (inf_u + kill_d) - ads * (bd + bu) * phage - decay * phage;
    // dR/dt
    dy[3] = inflow - yld * (mu_d * bd + mu_u * bu) - dilut * r;
    return dy;
}

fn rk4_step(y0: f64, y1: f64, y2: f64, y3: f64,
            p_base: u32, h: f64) -> array<f64, 4> {
    let z = h - h;
    let half = z + 0.5;
    let two  = z + 2.0;
    let h2 = h * half;

    let k1 = defense_deriv(y0, y1, y2, y3, p_base);
    let k2 = defense_deriv(
        y0 + h2 * k1[0], y1 + h2 * k1[1],
        y2 + h2 * k1[2], y3 + h2 * k1[3], p_base);
    let k3 = defense_deriv(
        y0 + h2 * k2[0], y1 + h2 * k2[1],
        y2 + h2 * k2[2], y3 + h2 * k2[3], p_base);
    let k4 = defense_deriv(
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
        y3 = fclamp(yn[3], cmin, cmax);
    }

    output_states[s_base + 0u] = y0;
    output_states[s_base + 1u] = y1;
    output_states[s_base + 2u] = y2;
    output_states[s_base + 3u] = y3;
}
