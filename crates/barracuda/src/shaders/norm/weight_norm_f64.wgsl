// SPDX-License-Identifier: AGPL-3.0-or-later
// weight_norm.wgsl - Weight Normalization (f64 canonical)
//
// Reparameterizes weights as: w = g * (v / ||v||)
// Reference: "Weight Normalization" by Salimans & Kingma (2016)
//
// Improves training speed and convergence

struct Params {
    num_weights: u32,
    dim: u32,  // Dimension to normalize over (0 = all)
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> v: array<f64>;             // Direction vectors
@group(0) @binding(1) var<storage, read> g: array<f64>;             // Magnitude scalars
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // Normalized weights
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_weights) {
        return;
    }

    if (params.dim == 0u) {
        // Normalize all weights together
        if (idx == 0u) {
            // Compute norm
            var norm_sq: f64 = 0.0;
            for (var i: u32 = 0u; i < params.num_weights; i = i + 1u) {
                norm_sq = norm_sq + v[i] * v[i];
            }
            let norm = sqrt_f64(norm_sq + 1e-12);

            // Apply: w = g * (v / ||v||)
            for (var i: u32 = 0u; i < params.num_weights; i = i + 1u) {
                output[i] = g[0] * v[i] / norm;
            }
        }
    } else {
        // Per-dimension normalization (simplified - would need grouping)
        output[idx] = v[idx];
    }
}
