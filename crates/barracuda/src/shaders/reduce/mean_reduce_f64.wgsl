// SPDX-License-Identifier: AGPL-3.0-or-later
// Mean Reduction (f64) - Parallel mean at double precision
//
// Computes sum of elements via tree reduction, then caller divides by n for mean.
// Algorithm: Tree reduction (work-efficient) with shared memory
//
// Use cases:
//   - Statistical analysis
//   - Batch normalization
//   - Scientific computing (accumulation accuracy)
//
// Notes:
//   - Outputs partial sums per workgroup (same as sum_reduce_f64)
//   - Final mean = sum(partials) / n on CPU or second pass

struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn mean_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        shared_data[tid] = input[gid];
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
