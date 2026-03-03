// Argmax Reduction - Find index of maximum value over all elements (f64 canonical)
// CUDA equivalent: thrust::reduce with argmax operation
// Algorithm: Tree reduction tracking both value and index
// Use cases: Global argmax computation

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;  // Partial results (indices)

struct Params {
    size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for values and indices
var<workgroup> shared_values: array<f64, 256>;
var<workgroup> shared_indices: array<u32, 256>;

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
    var index: u32;
    if (gid < params.size) {
        value = input[gid];
        index = gid;
    } else {
        // Initialize with -FLT_MAX for max reduction
        value = -3.402823e+38;
        index = 0xFFFFFFFFu; // Invalid index
    }
    shared_values[tid] = value;
    shared_indices[tid] = index;
    workgroupBarrier();

    // Tree reduction in shared memory, tracking indices
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (gid + stride) < params.size) {
            let a_val = shared_values[tid];
            let b_val = shared_values[tid + stride];
            let a_idx = shared_indices[tid];
            let b_idx = shared_indices[tid + stride];

            if (b_val > a_val) {
                shared_values[tid] = b_val;
                shared_indices[tid] = b_idx;
            } else {
                // Keep a_val and a_idx (already in place)
            }
        }
        workgroupBarrier();
    }

    // Write partial result (index of max value in this workgroup)
    if (tid == 0u) {
        output[workgroup_id.x] = shared_indices[0];
    }
}
