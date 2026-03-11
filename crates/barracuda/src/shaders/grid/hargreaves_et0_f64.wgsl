// SPDX-License-Identifier: AGPL-3.0-only
//
// hargreaves_et0_f64.wgsl — FAO Hargreaves reference evapotranspiration
//
// ET₀ = 0.0023 * Ra * (T_mean + 17.8) * (T_max - T_min)^0.5
// where Ra is extraterrestrial radiation (MJ/m²/day).
//
// Provenance: airSpring → ToadStool absorption

enable f64;

struct HargreavesParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> t_max: array<f64>;
@group(0) @binding(1) var<storage, read> t_min: array<f64>;
@group(0) @binding(2) var<storage, read> ra: array<f64>;
@group(0) @binding(3) var<storage, read_write> out: array<f64>;
@group(0) @binding(4) var<uniform> params: HargreavesParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let t_mean = (t_max[i] + t_min[i]) * 0.5;
    let t_range = max(t_max[i] - t_min[i], 0.0);
    out[i] = 0.0023 * ra[i] * (t_mean + 17.8) * pow(t_range, 0.5);
}
