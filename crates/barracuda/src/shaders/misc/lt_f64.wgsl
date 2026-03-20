// SPDX-License-Identifier: AGPL-3.0-or-later
// Less than - element-wise comparison (f64 canonical)
// output[i] = (a[i] < b[i]) ? 1.0 : 0.0

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&a)) {
        return;
    }
    
    if (a[idx] < b[idx]) {
        output[idx] = 1.0;
    } else {
        output[idx] = 0.0;
    }
}
