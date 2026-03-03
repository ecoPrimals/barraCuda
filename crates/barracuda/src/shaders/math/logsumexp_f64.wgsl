// SPDX-License-Identifier: AGPL-3.0-or-later
//
// logsumexp_f64.wgsl — Numerically stable log(sum(exp(x))) (f64)
//
// Two-pass: find max, then compute log(Σ exp(x_i - max)).
// Used in softmax, log-likelihood, numerical stability.
//
// Evolved from f32 → f64 for universal math library portability.

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 1u) {
        return;
    }

    var max_val: f64 = f64(-1.7976931348623157e+308);
    for (var i = 0u; i < metadata.size; i = i + 1u) {
        max_val = max(max_val, input[i]);
    }

    var sum: f64 = f64(0.0);
    for (var i = 0u; i < metadata.size; i = i + 1u) {
        sum = sum + exp(input[i] - max_val);
    }

    output[0] = max_val + log(sum);
}
