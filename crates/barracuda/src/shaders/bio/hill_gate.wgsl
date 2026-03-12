// SPDX-License-Identifier: AGPL-3.0-only
// hill_gate.wgsl — Two-input Hill AND gate (f32)
//
// neuralSpring absorption: regulatory network signal integration.
//
// Computes f(a, b) = vmax × H(a, K_a, n_a) × H(b, K_b, n_b)
// where H(x, K, n) = x^n / (K^n + x^n) is the Hill function.
//
// Two modes:
//   mode 0 — 1D paired:  a[i] paired with b[i], output[i]
//   mode 1 — 2D grid:    a[ix] crossed with b[iy], output[ix * ny + iy]
//
// Bindings:
//   0: input_a [N_a] f32
//   1: input_b [N_b] f32
//   2: output  [N_a × N_b] (mode 1) or [N_a] (mode 0) f32
//   3: params  uniform

struct HillGateParams {
    n_a:  u32,
    n_b:  u32,
    mode: u32,     // 0 = paired, 1 = grid
    _pad: u32,
    k_a:  f32,
    k_b:  f32,
    n_a_exp: f32,  // Hill exponent for input A
    n_b_exp: f32,  // Hill exponent for input B
    vmax: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0) var<storage, read>       input_a: array<f32>;
@group(0) @binding(1) var<storage, read>       input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  HillGateParams;

fn hill_f32(x: f32, k: f32, n: f32) -> f32 {
    let xc = max(x, 0.0);
    let xn = pow(xc, n);
    let kn = pow(max(k, 1e-20), n);
    return xn / (kn + xn);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (params.mode == 0u) {
        // Paired mode: one output per pair
        if (idx >= params.n_a) { return; }
        let ha = hill_f32(input_a[idx], params.k_a, params.n_a_exp);
        let hb = hill_f32(input_b[idx], params.k_b, params.n_b_exp);
        output[idx] = params.vmax * ha * hb;
    } else {
        // Grid mode: output[ix * n_b + iy]
        let total = params.n_a * params.n_b;
        if (idx >= total) { return; }
        let ix = idx / params.n_b;
        let iy = idx % params.n_b;
        let ha = hill_f32(input_a[ix], params.k_a, params.n_a_exp);
        let hb = hill_f32(input_b[iy], params.k_b, params.n_b_exp);
        output[idx] = params.vmax * ha * hb;
    }
}
