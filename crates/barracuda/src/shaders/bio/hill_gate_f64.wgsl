// SPDX-License-Identifier: AGPL-3.0-only
//
// hill_gate_f64.wgsl — Two-input Hill AND gate (f64)
//
// Computes f(a, b) = vmax × H(a, K_a, n_a) × H(b, K_b, n_b)
// where H(x, K, n) = x^n / (K^n + x^n) is the Hill function.
//
// Two modes:
//   mode 0 — 1D paired:  a[i] paired with b[i]
//   mode 1 — 2D grid:    a[ix] crossed with b[iy]
//
// Evolved from f32 → f64 for universal math library portability.

struct HillGateParams {
    n_a:  u32,
    n_b:  u32,
    mode: u32,
    _pad: u32,
    k_a:  f64,
    k_b:  f64,
    n_a_exp: f64,
    n_b_exp: f64,
    vmax: f64,
    _pad2: f64,
}

@group(0) @binding(0) var<storage, read>       input_a: array<f64>;
@group(0) @binding(1) var<storage, read>       input_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output:  array<f64>;
@group(0) @binding(3) var<uniform>             params:  HillGateParams;

fn hill_f64(x: f64, k: f64, n: f64) -> f64 {
    let xc = max(x, f64(0.0));
    let xn = pow(xc, n);
    let kn = pow(max(k, f64(1e-30)), n);
    return xn / (kn + xn);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (params.mode == 0u) {
        if (idx >= params.n_a) { return; }
        let ha = hill_f64(input_a[idx], params.k_a, params.n_a_exp);
        let hb = hill_f64(input_b[idx], params.k_b, params.n_b_exp);
        output[idx] = params.vmax * ha * hb;
    } else {
        let total = params.n_a * params.n_b;
        if (idx >= total) { return; }
        let ix = idx / params.n_b;
        let iy = idx % params.n_b;
        let ha = hill_f64(input_a[ix], params.k_a, params.n_a_exp);
        let hb = hill_f64(input_b[iy], params.k_b, params.n_b_exp);
        output[idx] = params.vmax * ha * hb;
    }
}
