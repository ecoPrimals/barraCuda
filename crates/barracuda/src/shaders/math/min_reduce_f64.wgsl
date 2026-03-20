// SPDX-License-Identifier: AGPL-3.0-or-later
// Min Reduction: Compute minimum over all elements (f64 canonical)
// CUDA equivalent: thrust::reduce with min operation
// Algorithm: Tree reduction (work-efficient)
// Use cases: Global min computation

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Partial results

struct Params {
    size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    var value: f64;
    if (gid < params.size) {
        value = input[gid];
    } else {
        // Initialize with FLT_MAX for min reduction
        value = 3.402823e+38;
    }
    shared_data[tid] = value;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (gid + stride) < params.size) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            shared_data[tid] = min(a, b);
        }
        workgroupBarrier();
    }

    // Write partial result
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
