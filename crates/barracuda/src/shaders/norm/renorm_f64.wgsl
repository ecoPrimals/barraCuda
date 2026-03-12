// SPDX-License-Identifier: AGPL-3.0-only
// Renorm - Renormalize with max norm constraint (f64 canonical)
// Clamps L2 norm of vectors to maximum value
// Used in training for gradient clipping alternative
//
// Algorithm:
// 1. Compute L2 norm: ||x||_2 = sqrt(Σ x_i^2)
// 2. If norm > max_norm: scale down: x = x * (max_norm / norm)
// 3. Else: keep unchanged

struct Params {
    outer: u32,      // Product of dimensions before norm_dim
    dim_size: u32,   // Size of dimension to normalize over
    inner: u32,      // Product of dimensions after norm_dim
    max_norm: f64,   // Maximum allowed norm
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
    let norm = sqrt_f64(norm_sq);

    // Compute scale factor (clamp norm to max_norm)
    var scale: f64 = 1.0;
    if (norm > params.max_norm) {
        scale = params.max_norm / norm;
    }

    // Apply renormalization
    for (var d = 0u; d < params.dim_size; d++) {
        let val_idx = o * params.dim_size * params.inner + d * params.inner + i;
        output[val_idx] = input[val_idx] * scale;
    }
}
