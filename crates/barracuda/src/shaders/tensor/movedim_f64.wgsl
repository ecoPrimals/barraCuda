// MoveDim - Move dimension to new position (complete implementation) (f64 canonical)
// Reorders tensor dimensions by moving a dimension from one position to another
//
// Example: movedim([B, C, H, W], source=1, destination=3) → [B, H, W, C]
//
// Algorithm:
// 1. Compute output strides from reordered shape
// 2. For each output position, compute corresponding input position
// 3. Copy value from input to output

struct Params {
    total_size: u32,
    num_dims: u32,
    source_dim: u32,
    dest_dim: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> input_shape: array<u32>;    // Original shape
@group(0) @binding(3) var<storage, read> output_shape: array<u32>;   // Reordered shape
@group(0) @binding(4) var<storage, read> input_strides: array<u32>;  // Input strides
@group(0) @binding(5) var<storage, read> output_strides: array<u32>; // Output strides
@group(0) @binding(6) var<storage, read> dim_mapping: array<u32>;    // Dimension mapping
@group(0) @binding(7) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.total_size) {
        return;
    }

    // Decompose output index into multi-dimensional coordinates
    var temp_idx = out_idx;
    var in_idx = 0u;

    for (var i = 0u; i < params.num_dims; i++) {
        // Get coordinate in output space
        let out_coord = temp_idx / output_strides[i];
        temp_idx = temp_idx % output_strides[i];

        // Map to input space using dimension mapping
        let in_dim = dim_mapping[i];
        in_idx += out_coord * input_strides[in_dim];
    }

    output[out_idx] = input[in_idx];
}
