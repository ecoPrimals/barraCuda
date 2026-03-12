// SPDX-License-Identifier: AGPL-3.0-only
// Element-wise subtraction: C = A - B

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = a[idx] - b[idx];
}
