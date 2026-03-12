// SPDX-License-Identifier: AGPL-3.0-only
// Permute - Reorder tensor dimensions (f64 canonical)
// Transposes tensor according to dimension permutation
//
// Example: permute([B, C, H, W], [0, 2, 3, 1]) → [B, H, W, C]
//
// Algorithm:
// For each output index, compute corresponding input index using permutation

struct Params {
    total_size: u32,
    num_dims: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> input_shape: array<u32>;    // Original shape
@group(0) @binding(3) var<storage, read> output_shape: array<u32>;   // Permuted shape
@group(0) @binding(4) var<storage, read> permutation: array<u32>;    // Dimension mapping
@group(0) @binding(5) var<storage, read> input_strides: array<u32>;  // Strides for input
@group(0) @binding(6) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_size) {
        return;
    }

    // Compute multi-dimensional output indices
    var temp_idx = out_idx;
    var in_idx = 0u;

    for (var i = 0u; i < params.num_dims; i++) {
        let dim_idx = temp_idx / output_shape[i + 1u]; // Division by stride
        temp_idx = temp_idx % output_shape[i + 1u];

        // Map to input dimension using permutation
        let in_dim = permutation[i];
        in_idx += dim_idx * input_strides[in_dim];
    }

    output[out_idx] = input[in_idx];
}
