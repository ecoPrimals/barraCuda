// SPDX-License-Identifier: AGPL-3.0-or-later
// pairwise_distance_f64.wgsl - Pairwise Distance (f64 canonical)
//
// Computes distance between pairs of vectors
// dist(x, y) = ||x - y||_p
//
// Supports L1, L2, and other p-norms

struct Params {
    num_pairs: u32,
    dim: u32,
    p: f64,         // p-norm (1.0 = L1, 2.0 = L2, etc.)
    epsilon: f64,   // For numerical stability
}

@group(0) @binding(0) var<storage, read> input1: array<f64>;       // [num_pairs, dim]
@group(0) @binding(1) var<storage, read> input2: array<f64>;       // [num_pairs, dim]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [num_pairs]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;

    if (pair_idx >= params.num_pairs) {
        return;
    }

    var dist: f64 = 0.0;

    for (var d: u32 = 0u; d < params.dim; d = d + 1u) {
        let idx = pair_idx * params.dim + d;
        let diff = abs(input1[idx] - input2[idx]);

        if (params.p == 1.0) {
            // L1 norm
            dist = dist + diff;
        } else if (params.p == 2.0) {
            // L2 norm (Euclidean)
            dist = dist + diff * diff;
        } else {
            // General p-norm
            dist = dist + pow_f64(diff + params.epsilon, params.p);
        }
    }

    // Take p-th root for p > 1
    if (params.p == 2.0) {
        dist = sqrt_f64(dist);
    } else if (params.p > 1.0 && params.p != 2.0) {
        dist = pow_f64(dist + params.epsilon, 1.0 / params.p);
    }

    output[pair_idx] = dist;
}
