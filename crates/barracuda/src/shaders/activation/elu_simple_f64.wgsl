// SPDX-License-Identifier: AGPL-3.0-only
// ELU (Exponential Linear Unit) - Simple version with alpha=1.0 (f64 canonical)
// Formula: ELU(x) = x if x > 0, else (exp(x) - 1)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&input) {
        return;
    }

    let x = input[idx];

    // ELU with alpha = 1.0
    if x > f64(0.0) {
        output[idx] = x;
    } else {
        output[idx] = exp_f64(x) - f64(1.0);
    }
}
