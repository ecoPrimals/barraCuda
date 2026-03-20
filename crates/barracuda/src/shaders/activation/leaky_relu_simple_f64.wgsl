// SPDX-License-Identifier: AGPL-3.0-or-later
// LeakyReLU - Simple version with alpha=0.01 (f64 canonical)
// Formula: LeakyReLU(x) = max(0.01x, x)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

const NEGATIVE_SLOPE: f64 = 0.01;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&input) {
        return;
    }

    let x = input[idx];

    if x > 0.0 {
        output[idx] = x;
    } else {
        output[idx] = NEGATIVE_SLOPE * x;
    }
}
