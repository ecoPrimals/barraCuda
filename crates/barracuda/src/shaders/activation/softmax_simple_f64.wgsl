// SPDX-License-Identifier: AGPL-3.0-or-later
// Simple Softmax for small tensors (single workgroup) (f64 canonical)
// Formula: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// `size` is passed as a uniform (logical tensor element count) rather than
// derived from `arrayLength(&input)`.  A pooled input buffer may be physically
// larger than the tensor's logical size; using arrayLength would include
// uninitialised padding elements in the reduction, breaking the normalisation.

struct Params {
    size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: f64;
var<workgroup> shared_sum: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = params.size;
    let idx = global_id.x;
    let tid = local_id.x;

    // Phase 1: Find max (numerically stable softmax)
    if (tid == 0u) {
        var max_val = input[0];
        for (var i = 1u; i < size; i = i + 1u) {
            max_val = max(max_val, input[i]);
        }
        shared_max = max_val;
    }
    workgroupBarrier();

    // Phase 2: Compute exp and sum
    if (tid == 0u) {
        var sum = f64(0.0);
        for (var i = 0u; i < size; i = i + 1u) {
            sum = sum + exp_f64(input[i] - shared_max);
        }
        shared_sum = sum;
    }
    workgroupBarrier();

    // Phase 3: Normalize
    if (idx < size) {
        output[idx] = exp_f64(input[idx] - shared_max) / shared_sum;
    }
}
