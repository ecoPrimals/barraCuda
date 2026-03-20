// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hessian_column.wgsl — Parallel central-difference Hessian column computation (f64 canonical)
//
// Computes one column j of the Hessian matrix using central differences:
//   H[i,j] = (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4*eps^2)
//
// Each thread computes H[i, col] for one row i. The dispatch provides the
// 4 function evaluations per row as pre-computed buffers (evaluated on CPU
// or in a prior GPU pass).
//
// Provenance: neuralSpring baseCamp V18 handoff (Feb 2026)
// Use case: loss landscape analysis, optimization, sensitivity analysis

struct Params {
    n: u32,
    col: u32,
    inv_4eps2: f64,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> f_pp: array<f64>;  // f(x + e_i + e_j)
@group(0) @binding(1) var<storage, read> f_pm: array<f64>;  // f(x + e_i - e_j)
@group(0) @binding(2) var<storage, read> f_mp: array<f64>;  // f(x - e_i + e_j)
@group(0) @binding(3) var<storage, read> f_mm: array<f64>;  // f(x - e_i - e_j)
@group(0) @binding(4) var<storage, read_write> hessian_col: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn hessian_column(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }

    let hij = (f_pp[i] - f_pm[i] - f_mp[i] + f_mm[i]) * params.inv_4eps2;
    hessian_col[i] = hij;
}
