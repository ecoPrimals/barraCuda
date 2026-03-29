// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Multi-shift CG: p_σ = ζ_σ * r + β_σ * p_σ (shifted direction update).
//
// beta_s_buf stores the ratio (ζ_new / ζ_curr) from the zeta kernel.
// The actual β_σ = ratio² * β_base, computed inline here.
//
// ζ_σ is the current ζ value (post-rotation, so zeta_curr[shift_idx]).
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

struct Params {
    n: u32,
    shift_idx: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> p: array<f64>;
@group(0) @binding(2) var<storage, read> r: array<f64>;
@group(0) @binding(3) var<storage, read> zeta_curr: array<f64>;
@group(0) @binding(4) var<storage, read> beta_ratio: array<f64>;
@group(0) @binding(5) var<storage, read> beta_base: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let i = idx;
    if (i >= params.n) { return; }

    let s = params.shift_idx;
    let zeta = zeta_curr[s];
    let ratio = beta_ratio[s];
    let b_base = beta_base[0];
    let beta = ratio * ratio * b_base;

    p[i] = zeta * r[i] + beta * p[i];
}
