// SPDX-License-Identifier: AGPL-3.0-or-later
// Flatten - Flatten tensor to specified dimensions (f64 canonical)
// Reshapes tensor by flattening specified range of dimensions
//
// Example: flatten([B, C, H, W], start=1, end=3) → [B, C*H*W]
//
// Algorithm:
// Simple memory copy (data is already contiguous)

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    // Flattening is just a reshape - data layout unchanged
    output[idx] = input[idx];
}
