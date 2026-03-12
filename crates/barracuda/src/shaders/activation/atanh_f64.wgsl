// SPDX-License-Identifier: AGPL-3.0-only
// Inverse hyperbolic tangent operation (f64 canonical)
// atanh(x) = 0.5 * ln((1 + x) / (1 - x))
// Defined for |x| < 1
//
// Uses elementwise_unary layout: binding 0 = input, binding 1 = output (no uniform).

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input)) {
        return;
    }

    let x = input[idx];

    // atanh is only defined for |x| < 1
    // atanh(x) = 0.5 * ln((1+x)/(1-x))
    output[idx] = f64(0.5) * log_f64((f64(1.0) + x) / (f64(1.0) - x));
}
