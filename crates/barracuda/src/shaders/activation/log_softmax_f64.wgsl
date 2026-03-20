// SPDX-License-Identifier: AGPL-3.0-or-later
// Log Softmax - Numerically stable log of softmax (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)
// - Numerically stable (avoid overflow/underflow)

struct Params {
    batch_size: u32,
    feature_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * params.feature_size;

    // Find max for numerical stability
    var max_val = input[base_idx];
    for (var i = 1u; i < params.feature_size; i = i + 1u) {
        max_val = max(max_val, input[base_idx + i]);
    }

    // Compute log(sum(exp(x - max)))
    var sum_exp = f64(0.0);
    for (var i = 0u; i < params.feature_size; i = i + 1u) {
        sum_exp = sum_exp + exp_f64(input[base_idx + i] - max_val);
    }
    let log_sum_exp = log_f64(sum_exp);

    // Compute log_softmax: x - max - log(sum(exp(x - max)))
    for (var i = 0u; i < params.feature_size; i = i + 1u) {
        let idx = base_idx + i;
        output[idx] = input[idx] - max_val - log_sum_exp;
    }
}
