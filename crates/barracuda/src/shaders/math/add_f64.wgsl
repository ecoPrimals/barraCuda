// SPDX-License-Identifier: AGPL-3.0-only
// SAXPY: Scaled Addition (c[i] = alpha * a[i] + b[i]) (f64 canonical)
//
// Efficient scaled vector addition.
// Used in: Residual connections, skip connections, tensor arithmetic

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> result: array<f64>;

struct Params {
    alpha: f64,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&result) {
        return;
    }
    
    result[idx] = params.alpha * a[idx] + b[idx];
}
