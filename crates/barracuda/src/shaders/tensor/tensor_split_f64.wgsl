// SPDX-License-Identifier: AGPL-3.0-or-later
// Tensor Split - Split tensor at specific indices (f64 canonical)
// Splits tensor into multiple chunks at specified positions
//
// Algorithm:
// For each output tensor, copy the appropriate slice based on split indices
// Similar to chunk but with variable-sized splits

struct Params {
    total_size: u32,
    num_splits: u32,
    split_dim: u32,
    dim_size: u32,       // Size of dimension being split
    inner_size: u32,     // Product of dimensions after split_dim
    outer_size: u32,     // Product of dimensions before split_dim
    split_start: u32,    // Start index in split dimension for this output
    split_size: u32,     // Size of this split along split dimension
    output_size: u32,    // Total output size for this split
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.output_size) {
        return;
    }

    // Decompose output index into coordinates
    // output_idx = outer * split_size * inner + split_coord * inner + inner_idx
    let outer = out_idx / (params.split_size * params.inner_size);
    let temp = out_idx % (params.split_size * params.inner_size);
    let split_coord = temp / params.inner_size;
    let inner = temp % params.inner_size;

    // Map to input index
    // input_coord = split_start + split_coord (within this split)
    let input_coord = params.split_start + split_coord;
    let in_idx = outer * params.dim_size * params.inner_size
                 + input_coord * params.inner_size
                 + inner;

    output[out_idx] = input[in_idx];
}
