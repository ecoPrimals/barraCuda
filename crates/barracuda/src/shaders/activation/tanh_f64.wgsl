// SPDX-License-Identifier: AGPL-3.0-only
// Hyperbolic tangent operation (f64 canonical)
// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let size = arrayLength(&input);

    if (idx >= size) {
        return;
    }

    let x = input[idx];
    output[idx] = tanh_f64(x);
}
