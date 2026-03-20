// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CG vector update: x += alpha * p, r -= alpha * ap.
// Alpha is read from a GPU buffer (no CPU upload needed).
//
// Absorbed from hotSpring lattice QCD (Feb 2026).

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> r: array<f64>;
@group(0) @binding(3) var<storage, read> p: array<f64>;
@group(0) @binding(4) var<storage, read> ap: array<f64>;
@group(0) @binding(5) var<storage, read> alpha: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let a = alpha[0];
    x[i] = x[i] + a * p[i];
    r[i] = r[i] - a * ap[i];
}
