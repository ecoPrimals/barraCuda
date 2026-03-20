// SPDX-License-Identifier: AGPL-3.0-or-later
// Take - Advanced indexing operation (gather) (f64 canonical)
// Gathers elements from input using index array
//
// Example: take([10, 20, 30, 40], [0, 2, 1]) → [10, 30, 20]
//
// Algorithm:
// For each output position, read index and gather from input

struct Params {
    output_size: u32,
    input_size: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.output_size) {
        return;
    }

    let in_idx = indices[out_idx];

    // Bounds check
    if (in_idx < params.input_size) {
        output[out_idx] = input[in_idx];
    } else {
        output[out_idx] = 0.0; // Out of bounds
    }
}
