// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Shifted CG vector update: x += alpha * p, r -= alpha * (ap + sigma * p).
// Alpha and sigma read from GPU buffers. Fuses the shift correction into
// the residual update to avoid an extra axpy dispatch.
//
// Absorbed from hotSpring lattice QCD (Mar 2026).

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
@group(0) @binding(6) var<storage, read> sigma: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let i = idx;
    if (i >= params.n) { return; }
    let a = alpha[0];
    let s = sigma[0];
    x[i] = x[i] + a * p[i];
    r[i] = r[i] - a * (ap[i] + s * p[i]);
}
