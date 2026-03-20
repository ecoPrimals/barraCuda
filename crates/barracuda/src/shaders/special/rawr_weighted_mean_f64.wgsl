// SPDX-License-Identifier: AGPL-3.0-or-later
// RAWR Weighted Mean (f64) — groundSpring V10/V54
//
// Reliability-Averaged Weighted Resampling: weighted mean with bootstrap confidence intervals.
// Each workgroup computes weighted mean for one bootstrap resample.
//
// Input: data array, weight array, resample indices (pre-generated, n_resamples * n)
// Output: bootstrap_means (one per resample)
//
// Weighted mean = Σ data[indices[i]] * weights[indices[i]] / Σ weights[indices[i]]
//
// f64 enabled by compile_shader_f64() preamble injection

struct Params {
    n: u32,
    n_resamples: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f64>;
@group(0) @binding(1) var<storage, read> weights: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> bootstrap_means: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resample_id = global_id.x;
    if (resample_id >= params.n_resamples) {
        return;
    }

    let n = params.n;
    let base = resample_id * n;

    var sum_wx: f64 = f64(0.0);
    var sum_w: f64 = f64(0.0);

    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let idx = indices[base + i];
        let d = data[idx];
        let w = weights[idx];
        sum_wx = sum_wx + d * w;
        sum_w = sum_w + w;
    }

    let mean = select(f64(0.0), sum_wx / sum_w, sum_w > f64(0.0));
    bootstrap_means[resample_id] = mean;
}
