// SPDX-License-Identifier: AGPL-3.0-or-later
// Element-wise minimum between tensor and scalar
// Universal compute via WGSL - works on any hardware

struct Params {
    size: u32,
    scalar: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Compute minimum of input and scalar
    output[idx] = min(input[idx], params.scalar);
}
