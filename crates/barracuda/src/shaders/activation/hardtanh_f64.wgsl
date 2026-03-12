// SPDX-License-Identifier: AGPL-3.0-only
// Hardtanh - Hard hyperbolic tangent (f64 canonical)
// Hardtanh: clamp(x, min_val, max_val)

struct Params {
    size: u32,
    min_val: f64,
    max_val: f64,
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

    let x = input[idx];
    output[idx] = clamp(x, params.min_val, params.max_val);
}
