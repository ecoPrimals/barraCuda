// SPDX-License-Identifier: AGPL-3.0-or-later
// Broadcast - expand tensor dimensions using NumPy-style broadcasting (f64 canonical)
//
// Maps each output element to its corresponding input element via
// stride-based indexing. Dimensions of size 1 in the input are broadcast
// (repeated) to match the output shape.
//
// Example: input [3, 1] → output [3, 4]
//   input_strides = [1, 0]  (stride 0 means broadcast this dim)
//   output_shape = [3, 4]
//
// Cross-domain: tensor arithmetic, batch processing, attention masks,
// physics field expansion.

struct Params {
    output_total: u32, // Total output elements
    ndim: u32,         // Number of dimensions (max 8)
    _padding: vec2<u32>,
}

// Shapes and strides stored as arrays (max 8 dimensions)
// input_strides[d] = 0 means broadcast (repeat) along dimension d
// input_strides[d] > 0 means normal stride

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> output_shape: array<u32>;   // [ndim]
@group(0) @binding(4) var<storage, read> input_strides: array<u32>;  // [ndim] (0 = broadcast)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.output_total) {
        return;
    }

    // Convert flat output index to multi-dimensional coordinates,
    // then map to input index using input_strides (0 = broadcast dim)
    var remaining = idx;
    var input_idx: u32 = 0u;

    // Row-major: iterate from last dim to first
    for (var d_rev: u32 = 0u; d_rev < params.ndim && d_rev < 8u; d_rev = d_rev + 1u) {
        let d = params.ndim - 1u - d_rev;
        let dim_size = output_shape[d];
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;

        // If input_strides[d] == 0, this dim is broadcast (coord ignored)
        input_idx = input_idx + coord * input_strides[d];
    }

    output[idx] = input[input_idx];
}
