// SPDX-License-Identifier: AGPL-3.0-or-later
// DotProduct_f64.wgsl — Compute inner product of two vectors (f64 canonical)
// CUDA equivalent: cublas::dot
// Use cases: Similarity, attention scores

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // Partial sums

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load and multiply
    var value: f64 = 0.0;
    if (gid < params.size) {
        value = a[gid] * b[gid];
    }
    shared_data[tid] = value;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write partial sum
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
