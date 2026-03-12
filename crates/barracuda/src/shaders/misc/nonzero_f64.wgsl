// SPDX-License-Identifier: AGPL-3.0-only
// NonZero - Find indices of non-zero elements (GPU parallel scan) (f64 canonical)
// Returns indices where tensor values are non-zero
//
// Algorithm:
// 1. Parallel scan to count non-zero elements and compute output positions
// 2. Compact write: each non-zero element writes its index to output

struct Params {
    input_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> prefix_sum: array<u32>;  // Precomputed prefix sum
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    // If element is non-zero, write its index to output
    if (input[idx] != 0.0) {
        // prefix_sum[idx] gives the output position for this element
        let out_pos = prefix_sum[idx]; // exclusive scan gives 0-based output index
        output[out_pos] = idx;
    }
}
