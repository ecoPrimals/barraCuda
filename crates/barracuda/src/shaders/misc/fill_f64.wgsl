// SPDX-License-Identifier: AGPL-3.0-or-later
// Fill - fill tensor with a constant value

struct FillParams {
    value: f64,
    _padding: vec3<f64>,
}

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = params.value;
}
