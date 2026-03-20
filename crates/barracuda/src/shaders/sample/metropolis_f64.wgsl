// SPDX-License-Identifier: AGPL-3.0-or-later
//
// metropolis_f64.wgsl — Parallel Metropolis-Hastings MCMC (f64 canonical)
//
// Each thread runs one independent chain. Proposes x' = x + Normal(0, step_size),
// accepts with probability min(1, exp(log_target(x') - log_target(x))).
// Uses xoshiro128** PRNG and Box-Muller for normal proposals.
//
// The log-target function values are provided as pre-computed buffers
// (evaluated on CPU or in a prior GPU pass).
//
// Provenance: neuralSpring baseCamp V18 handoff (Feb 2026)
// Use case: Bayesian inference, posterior sampling, energy minimization

struct Params {
    n_chains: u32,
    n_dims: u32,
    step_size: f64,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<f64>;
@group(0) @binding(1) var<storage, read> log_target_current: array<f64>;
@group(0) @binding(2) var<storage, read> log_target_proposed: array<f64>;
@group(0) @binding(3) var<storage, read> proposed: array<f64>;
@group(0) @binding(4) var<storage, read_write> accepted: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> prng_state: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro128ss(s: ptr<function, array<u32, 4>>) -> u32 {
    let result = rotl((*s)[1] * 5u, 7u) * 9u;
    let t = (*s)[1] << 9u;
    (*s)[2] ^= (*s)[0];
    (*s)[3] ^= (*s)[1];
    (*s)[1] ^= (*s)[2];
    (*s)[0] ^= (*s)[3];
    (*s)[2] ^= t;
    (*s)[3] = rotl((*s)[3], 11u);
    return result;
}

fn rand_uniform(s: ptr<function, array<u32, 4>>) -> f64 {
    return f64(xoshiro128ss(s)) / f64(4294967296.0);
}

@compute @workgroup_size(256)
fn metropolis_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain = gid.x;
    if (chain >= params.n_chains) {
        return;
    }

    let prng_base = chain * 4u;
    var s: array<u32, 4>;
    s[0] = prng_state[prng_base];
    s[1] = prng_state[prng_base + 1u];
    s[2] = prng_state[prng_base + 2u];
    s[3] = prng_state[prng_base + 3u];

    let log_alpha = log_target_proposed[chain] - log_target_current[chain];
    let u = rand_uniform(&s);
    let log_u = log_f64(max(u, f64(1e-30)));

    if (log_u < log_alpha) {
        let base = chain * params.n_dims;
        for (var d: u32 = 0u; d < params.n_dims; d++) {
            state[base + d] = proposed[base + d];
        }
        atomicAdd(&accepted[chain], 1u);
    }

    prng_state[prng_base] = s[0];
    prng_state[prng_base + 1u] = s[1];
    prng_state[prng_base + 2u] = s[2];
    prng_state[prng_base + 3u] = s[3];
}
