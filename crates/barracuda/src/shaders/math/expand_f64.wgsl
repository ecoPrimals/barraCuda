// SPDX-License-Identifier: AGPL-3.0-only
// Expand - Multi-dimensional broadcasting (NumPy-style) (f64 canonical)
// Broadcasts tensor to larger shape by expanding singleton dimensions
//
// Broadcasting rules:
// - Dimensions of size 1 can be broadcast to any size
// - Missing dimensions are added at the front with size 1
// - Example: (3, 1, 5) can broadcast to (3, 4, 5)
//
// Algorithm:
// For each output index, compute corresponding input index using strides
// When input dimension is 1, stride is 0 (always reads index 0)

struct Params {
    output_size: u32,
    num_dims: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> input_shape: array<u32>;    // Broadcasted input shape
@group(0) @binding(3) var<storage, read> output_shape: array<u32>;   // Target output shape
@group(0) @binding(4) var<storage, read> input_strides: array<u32>;  // Input strides (0 for broadcast dims)
@group(0) @binding(5) var<storage, read> output_strides: array<u32>; // Output strides
@group(0) @binding(6) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.output_size) {
        return;
    }
    
    // Decompose output index into multi-dimensional coordinates
    var temp_idx = out_idx;
    var in_idx = 0u;
    
    // Compute input index by mapping output coordinates through broadcasting
    for (var i = 0u; i < params.num_dims; i++) {
        // Get coordinate in output space
        let out_coord = temp_idx / output_strides[i];
        temp_idx = temp_idx % output_strides[i];
        
        // Map to input space:
        // - If input_shape[i] == 1, stride is 0, so always use coord 0
        // - Otherwise, use the output coordinate (may be modulo input_shape[i] if needed)
        let in_coord = select(out_coord % input_shape[i], 0u, input_shape[i] == 1u);
        in_idx += in_coord * input_strides[i];
    }
    
    output[out_idx] = input[in_idx];
}
