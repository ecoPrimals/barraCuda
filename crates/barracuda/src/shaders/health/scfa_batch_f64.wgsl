// SPDX-License-Identifier: AGPL-3.0-only
//
// scfa_batch_f64.wgsl — Batch SCFA production (element-wise Michaelis-Menten)
//
// Each thread computes acetate, propionate, butyrate for one fiber input.
// Output: 3 f64 values per element (acetate, propionate, butyrate).
//
// Absorbed from healthSpring V19 (Exp079, Exp085).
// Dispatch: (ceil(n_elements / 256), 1, 1)

enable f64;

struct Params {
    vmax_acetate: f64,
    km_acetate: f64,
    vmax_propionate: f64,
    km_propionate: f64,
    vmax_butyrate: f64,
    km_butyrate: f64,
    n_elements: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

fn michaelis_menten(vmax: f64, km: f64, substrate: f64) -> f64 {
    return vmax * substrate / (km + substrate);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_elements { return; }

    let fiber = input[idx];
    let base = idx * 3u;

    output[base]      = michaelis_menten(params.vmax_acetate, params.km_acetate, fiber);
    output[base + 1u] = michaelis_menten(params.vmax_propionate, params.km_propionate, fiber);
    output[base + 2u] = michaelis_menten(params.vmax_butyrate, params.km_butyrate, fiber);
}
