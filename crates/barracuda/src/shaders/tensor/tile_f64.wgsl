// SPDX-License-Identifier: AGPL-3.0-only
// Tile - Repeat tensor along dimensions (f64 canonical)
// Repeats the input tensor according to the repetition counts
//
// Example: tile([1, 2, 3], [2, 1]) → [[1, 2, 3], [1, 2, 3]]
//
// Algorithm:
// For each output position, compute corresponding input position using modulo

struct Params {
    total_size: u32,
    num_dims: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> input_shape: array<u32>;    // Original shape
@group(0) @binding(3) var<storage, read> output_shape: array<u32>;   // Tiled shape
@group(0) @binding(4) var<storage, read> input_strides: array<u32>;  // Input strides
@group(0) @binding(5) var<storage, read> output_strides: array<u32>; // Output strides
@group(0) @binding(6) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_size) {
        return;
    }

    // Compute multi-dimensional indices in output
    var temp_idx = out_idx;
    var in_idx = 0u;

    for (var i = 0u; i < params.num_dims; i++) {
        let out_dim_idx = temp_idx / output_strides[i];
        temp_idx = temp_idx % output_strides[i];

        // Map to input using modulo (wrapping)
        let in_dim_idx = out_dim_idx % input_shape[i];
        in_idx += in_dim_idx * input_strides[i];
    }

    output[out_idx] = input[in_idx];
}
