// SPDX-License-Identifier: AGPL-3.0-or-later
//
// CG vector update: p = r + beta * p.
// Beta is read from a GPU buffer (no CPU upload needed).
//
// Absorbed from hotSpring lattice QCD (Feb 2026).

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> p: array<f64>;
@group(0) @binding(2) var<storage, read> r: array<f64>;
@group(0) @binding(3) var<storage, read> beta: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = beta[0];
    p[i] = r[i] + b * p[i];
}
