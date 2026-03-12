// SPDX-License-Identifier: AGPL-3.0-only
// GLU - Gated Linear Unit (f64 canonical)
// glu(x) = a * sigmoid(b)
// where x is split into two halves: a and b

struct Params {
    size: u32,
    split_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let half_size = params.size / 2u;

    if (idx >= half_size) {
        return;
    }

    // First half: a
    let a = input[idx];

    // Second half: b
    let b = input[half_size + idx];

    // sigmoid(b)
    let sigmoid_b = f64(1.0) / (f64(1.0) + exp_f64(-b));

    // glu(x) = a * sigmoid(b)
    output[idx] = a * sigmoid_b;
}
