// SPDX-License-Identifier: AGPL-3.0-or-later
// bcs_bisection_f64.wgsl — BCS chemical potential bisection solver
//
// hotSpring absorption: nuclear physics BCS pairing.
//
// Finds the chemical potential λ such that Σ_k deg_k · v²_k(λ) = N
// where v²_k = 0.5 * (1 - (ε_k - λ) / sqrt((ε_k - λ)² + Δ²))
//
// Each thread handles one batch (nucleus) independently.
// Uses bisection on the monotone function f(λ) = Σ deg·v² - N.
//
// Validated to 6.2e-11 precision against analytical BCS solutions.
//
// Bindings:
//   0: config uniform { n_states, n_batch, max_iter, _pad, delta, tol, _pad2[2] }
//   1: eigenvalues [n_batch × n_states] f64
//   2: degeneracies [n_states] f64 — (2j+1)
//   3: target_N [n_batch] f64 — target particle number per batch
//   4: lower [n_batch] f64 — initial lower bound for λ
//   5: upper [n_batch] f64 — initial upper bound for λ
//   6: roots [n_batch] f64 — output λ values
//   7: iterations [n_batch] u32 — output iteration count

// f64 enabled by compile_shader_f64() preamble injection

struct BcsConfig {
    n_states: u32,
    n_batch:  u32,
    max_iter: u32,
    _pad:     u32,
    delta:    f64,
    tol:      f64,
    _pad2_0:  f64,
    _pad2_1:  f64,
}

@group(0) @binding(0) var<uniform>             config: BcsConfig;
@group(0) @binding(1) var<storage, read>       eigenvalues: array<f64>;
@group(0) @binding(2) var<storage, read>       degeneracies: array<f64>;
@group(0) @binding(3) var<storage, read>       target_N: array<f64>;
@group(0) @binding(4) var<storage, read>       lower: array<f64>;
@group(0) @binding(5) var<storage, read>       upper: array<f64>;
@group(0) @binding(6) var<storage, read_write> roots: array<f64>;
@group(0) @binding(7) var<storage, read_write> iterations: array<u32>;

fn bcs_particle_number(lambda: f64, batch: u32, ns: u32, delta: f64) -> f64 {
    let zero = delta - delta;
    let half = zero + 0.5;
    let one = zero + 1.0;
    let d2 = delta * delta;
    let base = batch * ns;

    var total = zero;
    for (var k = 0u; k < ns; k = k + 1u) {
        let xi = eigenvalues[base + k] - lambda;
        let denom = sqrt(xi * xi + d2);
        var v2 = half;
        if (denom > zero + 1e-30) {
            v2 = half * (one - xi / denom);
        }
        total = total + degeneracies[k] * v2;
    }
    return total;
}

@compute @workgroup_size(256)
fn bisect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if (batch >= config.n_batch) { return; }

    let ns = config.n_states;
    let zero = config.delta - config.delta;
    let half = zero + 0.5;
    let target = target_N[batch];

    var lo = lower[batch];
    var hi = upper[batch];
    var iter = 0u;

    for (var i = 0u; i < config.max_iter; i = i + 1u) {
        let mid = (lo + hi) * half;
        let n_mid = bcs_particle_number(mid, batch, ns, config.delta);

        if (n_mid < target) {
            lo = mid;
        } else {
            hi = mid;
        }

        iter = i + 1u;
        if ((hi - lo) < config.tol) {
            break;
        }
    }

    roots[batch] = (lo + hi) * half;
    iterations[batch] = iter;
}
