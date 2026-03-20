// SPDX-License-Identifier: AGPL-3.0-or-later
// Stack - Stack tensors along new dimension (f64 canonical)
// Concatenates tensors along a new axis
//
// Example: stack([A, B, C], dim=0) → [[A], [B], [C]]
//
// Algorithm:
// Output shape: insert new dimension at position `dim`
// Copy each tensor into its slice of the output

struct Params {
    num_tensors: u32,
    tensor_size: u32,      // Size of each tensor (product of all dims)
    output_size: u32,      // Total output size
    stack_dim: u32,        // Dimension to stack along
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;      // All tensors concatenated
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.output_size) {
        return;
    }

    // Determine which tensor and which position within tensor
    let tensor_idx = idx / params.tensor_size;
    let within_tensor = idx % params.tensor_size;

    if (tensor_idx < params.num_tensors) {
        let input_idx = tensor_idx * params.tensor_size + within_tensor;
        output[idx] = input[input_idx];
    }
}
