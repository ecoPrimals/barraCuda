// SPDX-License-Identifier: AGPL-3.0-only
// Split operation - Split tensor along a dimension (inverse of Concat) (f64 canonical)
// Copies portions of input to multiple outputs

struct SplitParams {
    total_size: u32,
    split_point: u32,  // Where to split
    _pad: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output1: array<f64>;
@group(0) @binding(2) var<storage, read_write> output2: array<f64>;
@group(0) @binding(3) var<uniform> params: SplitParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.total_size) {
        return;
    }
    
    // Split at split_point
    if (idx < params.split_point) {
        output1[idx] = input[idx];
    } else {
        output2[idx - params.split_point] = input[idx];
    }
}
