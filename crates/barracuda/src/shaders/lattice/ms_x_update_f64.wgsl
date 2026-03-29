// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Multi-shift CG: x_σ += α_σ * p_σ (shifted solution update).
// The scalar α_σ is read from a GPU buffer at index [shift_idx].
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

struct Params {
    n: u32,
    shift_idx: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f64>;
@group(0) @binding(2) var<storage, read> p: array<f64>;
@group(0) @binding(3) var<storage, read> alpha_s: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let i = idx;
    if (i >= params.n) { return; }
    let a = alpha_s[params.shift_idx];
    x[i] = x[i] + a * p[i];
}
