// SPDX-License-Identifier: AGPL-3.0-only
// Softmax: Numerically stable softmax activation (f64 canonical)
// CUDA equivalent: cudnn::Softmax
// Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Use cases: Classification output, attention weights

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<storage, read_write> max_val: array<f64>;  // For stability
@group(0) @binding(3) var<storage, read_write> sum_val: array<f64>;

struct Params {
    size: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for reductions (must be at module scope)
var<workgroup> shared_max: array<f64, 256>;
var<workgroup> shared_sum: array<f64, 256>;

// Pass 1: Find max value
@compute @workgroup_size(256)
fn find_max(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    var value: f64 = f64(-3.402823e+38);  // -FLT_MAX
    if (gid < params.size) {
        value = input[gid];
    }
    shared_max[tid] = value;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        max_val[workgroup_id.x] = shared_max[0];
    }
}

// Pass 2: Compute exp(x - max) and sum
@compute @workgroup_size(256)
fn compute_exp_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Assume max_val[0] contains the global max (computed on CPU or via multi-pass)
    let global_max = max_val[0];

    var value: f64 = f64(0.0);
    if (gid < params.size) {
        let exp_val = exp_f64(input[gid] - global_max);
        output[gid] = exp_val;  // Store exp values
        value = exp_val;
    }
    shared_sum[tid] = value;
    workgroupBarrier();

    // Tree reduction for sum
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        sum_val[workgroup_id.x] = shared_sum[0];
    }
}

// Pass 3: Divide by sum
@compute @workgroup_size(256)
fn normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let gid = global_id.x;

    if (gid < params.size) {
        // Assume sum_val[0] contains the global sum
        output[gid] = output[gid] / sum_val[0];
    }
}
