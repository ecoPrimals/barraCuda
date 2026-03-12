// SPDX-License-Identifier: AGPL-3.0-only
// VPC (Visual Predictive Check) Monte Carlo simulation — GPU kernel.
//
// Each thread simulates one individual under the population PK model with
// random inter-individual variability (IIV). Embarrassingly parallel.
//
// Uses a simple LCG PRNG for Box-Muller normal sampling.
//
// Provenance: healthSpring V14 → barraCuda absorption (Mar 2026)

struct VpcConfig {
    n_simulations: u32,
    n_time_points: u32,
    n_compartments: u32,
    n_steps_per_interval: u32,
    seed_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> config: VpcConfig;
@group(0) @binding(1) var<storage, read> pop_params: array<f64>;
@group(0) @binding(2) var<storage, read> omega_diag: array<f64>;
@group(0) @binding(3) var<storage, read> time_points: array<f64>;
@group(0) @binding(4) var<storage, read_write> concentrations: array<f64>;

fn lcg_next(state: ptr<function, u32>) -> u32 {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

fn uniform_01(state: ptr<function, u32>) -> f64 {
    let u = lcg_next(state);
    return f64(u) / 4294967295.0;
}

fn normal_sample(state: ptr<function, u32>) -> f64 {
    let u1 = max(uniform_01(state), 1e-10);
    let u2 = uniform_01(state);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sim = gid.x;
    if sim >= config.n_simulations { return; }

    var rng_state: u32 = config.seed_base + sim * 7919u;

    // Sample individual parameters: θ_i = θ_pop · exp(η_i)
    // where η_i ~ N(0, ω²)
    let n_params = 3u; // CL, V, ka (typical one-compartment)
    var theta: array<f64, 3>;
    for (var p: u32 = 0u; p < n_params; p = p + 1u) {
        let eta = normal_sample(&rng_state) * sqrt(omega_diag[p]);
        theta[p] = pop_params[p] * exp(eta);
    }

    let cl = theta[0];
    let v = theta[1];
    let ka = theta[2];
    let ke = cl / v;

    // Simulate one-compartment oral PK: dA_gut/dt = -ka·A_gut, dA_cent/dt = ka·A_gut - ke·A_cent
    var a_gut: f64 = 1.0; // unit dose
    var a_cent: f64 = 0.0;

    let out_base = sim * config.n_time_points;
    var t_current: f64 = 0.0;

    for (var tp: u32 = 0u; tp < config.n_time_points; tp = tp + 1u) {
        let t_target = time_points[tp];
        let dt = (t_target - t_current) / f64(config.n_steps_per_interval);

        for (var s: u32 = 0u; s < config.n_steps_per_interval; s = s + 1u) {
            // RK4 step
            let k1_gut = -ka * a_gut;
            let k1_cent = ka * a_gut - ke * a_cent;

            let ag2 = a_gut + 0.5 * dt * k1_gut;
            let ac2 = a_cent + 0.5 * dt * k1_cent;
            let k2_gut = -ka * ag2;
            let k2_cent = ka * ag2 - ke * ac2;

            let ag3 = a_gut + 0.5 * dt * k2_gut;
            let ac3 = a_cent + 0.5 * dt * k2_cent;
            let k3_gut = -ka * ag3;
            let k3_cent = ka * ag3 - ke * ac3;

            let ag4 = a_gut + dt * k3_gut;
            let ac4 = a_cent + dt * k3_cent;
            let k4_gut = -ka * ag4;
            let k4_cent = ka * ag4 - ke * ac4;

            a_gut = a_gut + dt / 6.0 * (k1_gut + 2.0 * k2_gut + 2.0 * k3_gut + k4_gut);
            a_cent = a_cent + dt / 6.0 * (k1_cent + 2.0 * k2_cent + 2.0 * k3_cent + k4_cent);
        }

        concentrations[out_base + tp] = a_cent / v;
        t_current = t_target;
    }
}
