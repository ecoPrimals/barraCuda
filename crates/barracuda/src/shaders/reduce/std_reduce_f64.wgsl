// SPDX-License-Identifier: AGPL-3.0-only
// Standard Deviation Reduction (f64) - Two-pass std at double precision
//
// Input: array<f64>, params: {n: u32, mean: f64}
// Each thread computes (x - mean)^2, then tree reduce sum.
// Output: partial sums of squared deviations per workgroup.
// Final std = sqrt(sum(partials) / n) on CPU or second pass.
//
// Use cases:
//   - Statistical analysis
//   - Error estimation
//   - Scientific measurements

struct StdParams {
    n: u32,
    _pad1: u32,
    mean: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: StdParams;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn std_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.n) {
        let diff = input[gid] - params.mean;
        shared_data[tid] = diff * diff;
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
