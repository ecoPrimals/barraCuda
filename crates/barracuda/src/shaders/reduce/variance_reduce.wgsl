// SPDX-License-Identifier: AGPL-3.0-only
// Variance Reduction: Compute variance over all elements
// Algorithm: Two-pass tree reduction
// Pass 1: Compute mean using tree reduction
// Pass 2: Compute variance using tree reduction with mean
// Var(X) = E[(X - μ)²]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;  // Partial results

struct Params {
    size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    var value: f32;
    if (gid < params.size) {
        value = input[gid];
    } else {
        // Initialize with 0 for sum reduction
        value = 0.0;
    }
    shared_data[tid] = value;
    workgroupBarrier();
    
    // Tree reduction in shared memory (computing sum for mean)
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (gid + stride) < params.size) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            shared_data[tid] = a + b;
        }
        workgroupBarrier();
    }
    
    // Write partial sum result
    // Note: Mean will be computed on CPU after reading partial sums
    // Variance will be computed in a second pass
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
