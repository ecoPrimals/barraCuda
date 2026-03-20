// SPDX-License-Identifier: AGPL-3.0-or-later
// Normalize - L2 normalization along dimension (f64 canonical)
// Normalizes vectors to unit length: x_normalized = x / ||x||_2
//
// Algorithm:
// 1. Compute L2 norm along specified dimension: ||x||_2 = sqrt(Σ x_i^2)
// 2. Divide by norm (with epsilon for numerical stability)

struct Params {
    outer: u32,      // Product of dimensions before norm_dim
    dim_size: u32,   // Size of dimension to normalize over
    inner: u32,      // Product of dimensions after norm_dim
    epsilon: f64,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.outer * params.inner;

    if (idx >= total) {
        return;
    }

    let o = idx / params.inner;
    let i = idx % params.inner;

    // Compute L2 norm
    var norm_sq: f64 = 0.0;
    for (var d = 0u; d < params.dim_size; d++) {
        let val_idx = o * params.dim_size * params.inner + d * params.inner + i;
        let val = input[val_idx];
        norm_sq = norm_sq + val * val;
    }
    let norm = sqrt_f64(norm_sq) + params.epsilon;

    // Normalize
    for (var d = 0u; d < params.dim_size; d++) {
        let val_idx = o * params.dim_size * params.inner + d * params.inner + i;
        output[val_idx] = input[val_idx] / norm;
    }
}
