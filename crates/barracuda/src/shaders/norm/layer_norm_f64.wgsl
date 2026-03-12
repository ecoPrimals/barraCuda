// SPDX-License-Identifier: AGPL-3.0-only
// Layer Normalization - Normalize along feature dimension (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    size: u32,
    feature_size: u32,
    epsilon: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let num_batches = params.size / params.feature_size;

    if (batch_idx >= num_batches) {
        return;
    }

    let base_idx = batch_idx * params.feature_size;

    // Compute mean
    var sum: f64 = 0.0;
    for (var i = 0u; i < params.feature_size; i = i + 1u) {
        sum = sum + input[base_idx + i];
    }
    let mean = sum / f64(params.feature_size);

    // Compute variance
    var var_sum: f64 = 0.0;
    for (var i = 0u; i < params.feature_size; i = i + 1u) {
        let diff = input[base_idx + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f64(params.feature_size);

    // Normalize
    let std_dev = sqrt_f64(variance + params.epsilon);
    for (var i = 0u; i < params.feature_size; i = i + 1u) {
        let idx = base_idx + i;
        output[idx] = (input[idx] - mean) / std_dev;
    }
}
