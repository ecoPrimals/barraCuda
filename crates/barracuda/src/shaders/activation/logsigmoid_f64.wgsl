// SPDX-License-Identifier: AGPL-3.0-or-later
// LogSigmoid - Logarithm of sigmoid (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

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
    // LogSigmoid(x) = log(sigmoid(x)) = log(1 / (1 + exp(-x)))
    // For numerical stability: -log(1 + exp(-x)) for x >= 0, x - log(1 + exp(x)) for x < 0
    if (x >= f64(0.0)) {
        output[idx] = -log_f64(f64(1.0) + exp_f64(-x));
    } else {
        output[idx] = x - log_f64(f64(1.0) + exp_f64(x));
    }
}
