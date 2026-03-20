// SPDX-License-Identifier: AGPL-3.0-or-later
// LayerNorm Mean+Variance Computation (Single Pass) - f64 canonical
//
// **2-DISPATCH LAYERNORM - DISPATCH 1: COMPUTE MEAN AND VARIANCE TOGETHER**
//
// Algorithm (Welford's two-pass):
//   1. First scan: Compute mean using Welford's algorithm
//   2. Second scan: Compute variance from mean

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> stats: array<f64>;  // [mean, variance]

struct Params {
    size: u32,
    epsilon: f64,
}
@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for reduction
var<workgroup> shared_sum: array<f64, 256>;
var<workgroup> shared_sq_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let total_threads = 256u;

    var local_sum: f64 = 0.0;

    for (var i = global_id.x; i < params.size; i = i + total_threads) {
        local_sum = local_sum + input[i];
    }

    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    var mean: f64;
    if (tid == 0u) {
        mean = shared_sum[0] / f64(params.size);
        stats[0] = mean;
    }
    workgroupBarrier();

    mean = stats[0];

    var local_sq_sum: f64 = 0.0;

    for (var i = global_id.x; i < params.size; i = i + total_threads) {
        let diff = input[i] - mean;
        local_sq_sum = local_sq_sum + diff * diff;
    }

    shared_sq_sum[tid] = local_sq_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let variance = shared_sq_sum[0] / f64(params.size);
        stats[1] = variance;
    }
}
