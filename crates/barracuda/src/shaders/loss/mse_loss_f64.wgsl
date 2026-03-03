// SPDX-License-Identifier: AGPL-3.0-only
// MSE Loss (f64) - Mean Squared Error at double precision
//
// output = sum((pred - target)^2) / n
// Uses tree reduction for accumulation accuracy in scientific ML.
// Outputs partial sums per workgroup; caller sums and divides by n for final scalar.

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = arrayLength(&predictions);

    if (gid < n) {
        let diff = predictions[gid] - targets[gid];
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
