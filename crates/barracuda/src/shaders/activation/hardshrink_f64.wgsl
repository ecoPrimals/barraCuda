// SPDX-License-Identifier: AGPL-3.0-or-later
// Hardshrink - Hard shrinkage activation function (f64 canonical)
// Hardshrink: f(x) = x if |x| > lambda, 0 otherwise

struct Params {
    size: u32,
    lambda: f64,
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

    if (abs(x) > params.lambda) {
        output[idx] = x;
    } else {
        output[idx] = 0.0;
    }
}
