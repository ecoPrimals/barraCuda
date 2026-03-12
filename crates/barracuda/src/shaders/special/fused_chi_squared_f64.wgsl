// SPDX-License-Identifier: AGPL-3.0-only
// Fused Chi-Squared Test (f64) — neuralSpring V24
//
// Computes observed vs expected, chi-squared statistic, and p-value in a single GPU dispatch.
// χ² = Σ (observed - expected)² / expected
// p-value = 1 - P(k/2, χ²/2) where P is regularized lower incomplete gamma
// Uses same gamma functions as chi_squared_f64.wgsl.
//
// Input: observed, expected arrays (same length n)
// Output: partial chi² sums per workgroup, then reduced to scalar
// p-value computed on CPU from final χ² and df = n-1
//
// f64 enabled by compile_shader_f64() preamble injection

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> observed: array<f64>;
@group(0) @binding(1) var<storage, read> expected: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

// Chi-squared partial sum: output[wg_id] = Σ (obs[i]-exp[i])²/exp[i] for this workgroup's range
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.n) {
        let o = observed[gid];
        let e = expected[gid];
        let term = select(f64(0.0), (o - e) * (o - e) / e, e > f64(0.0));
        shared_data[tid] = term;
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
