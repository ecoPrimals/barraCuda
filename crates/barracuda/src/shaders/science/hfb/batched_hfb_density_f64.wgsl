// SPDX-License-Identifier: AGPL-3.0-or-later
// batched_hfb_density_f64.wgsl — BCS pairing + density construction for spherical HFB
//
// hotSpring absorption: nuclear physics GPU-resident SCF.
//
// Three entry points (kernels):
//   1. bcs_occupations — Compute BCS v² from eigenvalues and gap parameter
//   2. compute_density — Sum ρ(r) = Σ_k deg_k · v²_k · |φ_k(r)|² / (4π)
//   3. mix_density     — Linear density mixing for SCF convergence
//
// Physics:
//   v²_k = 0.5 * (1 - (ε_k - λ) / sqrt((ε_k - λ)² + Δ²))
//   ρ(r) = Σ_k (2j_k + 1) · v²_k · |φ_k(r)|² / (4πr²)
//
// Bindings:
//   0: params       uniform { n_states, n_grid, n_batch, mix_alpha: f64, delta: f64, lambda: f64 }
//   1: eigenvalues  [n_batch × n_states] f64
//   2: degeneracies [n_states] f64 — (2j+1) for each state
//   3: wf           [n_batch × n_states × n_grid] f64 — radial wavefunctions
//   4: v2_out       [n_batch × n_states] f64 — output BCS occupation probabilities
//   5: rho_out      [n_batch × n_grid] f64 — output density
//   6: rho_prev     [n_batch × n_grid] f64 — previous density (for mixing)
//   7: r_grid       [n_grid] f64

// f64 enabled by compile_shader_f64() preamble injection

struct DensityParams {
    n_states:  u32,
    n_grid:    u32,
    n_batch:   u32,
    _pad:      u32,
    mix_alpha: f64,
    delta:     f64,
    lambda:    f64,
}

@group(0) @binding(0) var<uniform>             params: DensityParams;
@group(0) @binding(1) var<storage, read>       eigenvalues: array<f64>;
@group(0) @binding(2) var<storage, read>       degeneracies: array<f64>;
@group(0) @binding(3) var<storage, read>       wf: array<f64>;
@group(0) @binding(4) var<storage, read_write> v2_out: array<f64>;
@group(0) @binding(5) var<storage, read_write> rho_out: array<f64>;
@group(0) @binding(6) var<storage, read>       rho_prev: array<f64>;
@group(0) @binding(7) var<storage, read>       r_grid: array<f64>;

@compute @workgroup_size(256)
fn bcs_occupations(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let ns = params.n_states;
    let total = params.n_batch * ns;
    if (idx >= total) { return; }

    let zero = params.delta - params.delta;
    let half = zero + 0.5;
    let one = zero + 1.0;

    let eps = eigenvalues[idx];
    let xi = eps - params.lambda;
    let d2 = params.delta * params.delta;
    let denom = sqrt(xi * xi + d2);

    var v2 = half;
    if (denom > zero + 1e-30) {
        v2 = half * (one - xi / denom);
    }

    v2_out[idx] = clamp(v2, zero, one);
}

@compute @workgroup_size(256)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let ng = params.n_grid;
    let total = params.n_batch * ng;
    if (idx >= total) { return; }

    let batch = idx / ng;
    let ir = idx % ng;
    let ns = params.n_states;

    let zero = r_grid[0] - r_grid[0];
    let four_pi = zero + 12.566370614359172;
    let r = r_grid[ir];
    let r2 = r * r;

    var rho = zero;
    for (var k = 0u; k < ns; k = k + 1u) {
        let v2 = v2_out[batch * ns + k];
        let phi = wf[batch * ns * ng + k * ng + ir];
        let deg = degeneracies[k];
        rho = rho + deg * v2 * phi * phi;
    }

    // Divide by 4πr² (spherical shell normalization)
    if (r2 > zero + 1e-30) {
        rho = rho / (four_pi * r2);
    }

    rho_out[idx] = rho;
}

@compute @workgroup_size(256)
fn mix_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n_batch * params.n_grid;
    if (idx >= total) { return; }

    let zero = params.mix_alpha - params.mix_alpha;
    let one = zero + 1.0;
    let alpha = params.mix_alpha;

    rho_out[idx] = alpha * rho_out[idx] + (one - alpha) * rho_prev[idx];
}
