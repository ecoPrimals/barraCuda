// SPDX-License-Identifier: AGPL-3.0-only
// RMS Normalization (Root Mean Square Normalization) (f64 canonical)
// Simpler alternative to LayerNorm used in modern transformers
//
// RMSNorm(x) = x / sqrt(mean(x²) + epsilon) * gamma
//
// Key difference from LayerNorm: No mean subtraction, only RMS scaling
// This simplification reduces computation while maintaining effectiveness
//
// Used in: LLaMA, GPT-NeoX, T5, modern large language models
// Benefits: Faster than LayerNorm, similar performance, simpler computation

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // Scale parameters
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    batch_size: u32,
    feature_size: u32,
    epsilon: f64,
    _padding: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.batch_size {
        return;
    }

    // Compute RMS (root mean square) for this sample
    var sum_sq: f64 = 0.0;
    let base_idx = idx * params.feature_size;

    for (var i: u32 = 0u; i < params.feature_size; i = i + 1u) {
        let x = input[base_idx + i];
        sum_sq = sum_sq + x * x;
    }

    let rms = sqrt_f64(sum_sq / f64(params.feature_size) + params.epsilon);

    // Normalize and scale
    for (var i: u32 = 0u; i < params.feature_size; i = i + 1u) {
        let x = input[base_idx + i];
        output[base_idx + i] = (x / rms) * gamma[i];
    }
}
