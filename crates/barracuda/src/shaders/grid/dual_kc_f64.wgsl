// SPDX-License-Identifier: AGPL-3.0-only
//
// dual_kc_f64.wgsl — FAO dual crop coefficient
//
// ETc = (Kcb * Ks + Ke) * ET₀
// where Kcb is basal crop coefficient, Ks is water stress coefficient,
// Ke is evaporation coefficient.
//
// Provenance: airSpring → ToadStool absorption

enable f64;

struct DualKcParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> kcb: array<f64>;
@group(0) @binding(1) var<storage, read> ks: array<f64>;
@group(0) @binding(2) var<storage, read> ke: array<f64>;
@group(0) @binding(3) var<storage, read> et0: array<f64>;
@group(0) @binding(4) var<storage, read_write> out: array<f64>;
@group(0) @binding(5) var<uniform> params: DualKcParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    out[i] = (kcb[i] * ks[i] + ke[i]) * et0[i];
}
