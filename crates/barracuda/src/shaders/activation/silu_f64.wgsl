// SPDX-License-Identifier: AGPL-3.0-or-later
// SiLU - Sigmoid Linear Unit (same as Swish) (f64 canonical)
// silu(x) = x * sigmoid(x) = x / (1 + e^(-x))

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];

    // sigmoid(x) = 1 / (1 + e^(-x))
    let sigmoid_val = f64(1.0) / (f64(1.0) + exp_f64(-x));

    // silu(x) = x * sigmoid(x)
    let result = x * sigmoid_val;

    output[idx] = result;
}
