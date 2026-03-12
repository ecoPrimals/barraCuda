// SPDX-License-Identifier: AGPL-3.0-only
// LayerNorm Statistics Computation (Dispatch 1 of 2) - f64 canonical
//
// **2-DISPATCH FUSED LAYERNORM - PART 1: STATISTICS**
//
// This shader computes partial mean and variance per workgroup.

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> partial_stats: array<f64>;  // [mean, m2, count] per workgroup
@group(0) @binding(2) var<storage, read_write> global_stats: array<f64>;   // [mean, variance]

struct Params {
    size: u32,
    epsilon: f64,
    num_workgroups: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_mean: array<f64, 256>;
var<workgroup> shared_m2: array<f64, 256>;
var<workgroup> shared_count: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let tid = local_id.x;
    let wg_id = workgroup_id.x;
    let total_threads = 256u * num_workgroups.x;

    var local_mean: f64 = 0.0;
    var local_m2: f64 = 0.0;
    var local_count: u32 = 0u;

    for (var i = global_id.x; i < params.size; i = i + total_threads) {
        let value = input[i];
        local_count = local_count + 1u;

        let delta = value - local_mean;
        local_mean = local_mean + delta / f64(local_count);
        let delta2 = value - local_mean;
        local_m2 = local_m2 + delta * delta2;
    }

    shared_mean[tid] = local_mean;
    shared_m2[tid] = local_m2;
    shared_count[tid] = local_count;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (tid + stride) < 256u) {
            let count_a = shared_count[tid];
            let count_b = shared_count[tid + stride];
            let total_count = count_a + count_b;

            if (total_count > 0u) {
                let mean_a = shared_mean[tid];
                let mean_b = shared_mean[tid + stride];
                let m2_a = shared_m2[tid];
                let m2_b = shared_m2[tid + stride];

                let delta = mean_b - mean_a;
                let combined_mean = mean_a + delta * f64(count_b) / f64(total_count);

                let combined_m2 = m2_a + m2_b + delta * delta * f64(count_a) * f64(count_b) / f64(total_count);

                shared_mean[tid] = combined_mean;
                shared_m2[tid] = combined_m2;
                shared_count[tid] = total_count;
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let base = wg_id * 3u;
        partial_stats[base] = shared_mean[0];
        partial_stats[base + 1u] = shared_m2[0];
        partial_stats[base + 2u] = f64(shared_count[0]);
    }
}
