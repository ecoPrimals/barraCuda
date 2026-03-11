// SPDX-License-Identifier: AGPL-3.0-only
//
// cdist_f64.wgsl — Pairwise distance computation (f64)
//
// Computes distances between all pairs of vectors from two sets.
// Each thread handles one (i, j) pair.
//
// Metrics: 0=Euclidean, 1=Manhattan, 2=Cosine distance
//
// Evolved from f32 cdist.wgsl for universal math library portability.

struct Params {
    m: u32,
    n: u32,
    d: u32,
    metric: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f64>;
@group(0) @binding(1) var<storage, read> input_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= params.m || j >= params.n) {
        return;
    }

    var dist: f64 = f64(0.0);

    if (params.metric == 0u) {
        // Euclidean (L2)
        var sum_sq: f64 = f64(0.0);
        for (var k = 0u; k < params.d; k = k + 1u) {
            let diff = input_a[i * params.d + k] - input_b[j * params.d + k];
            sum_sq = sum_sq + diff * diff;
        }
        dist = sqrt(sum_sq);
    } else if (params.metric == 1u) {
        // Manhattan (L1)
        var sum_abs: f64 = f64(0.0);
        for (var k = 0u; k < params.d; k = k + 1u) {
            let diff = input_a[i * params.d + k] - input_b[j * params.d + k];
            sum_abs = sum_abs + abs(diff);
        }
        dist = sum_abs;
    } else if (params.metric == 2u) {
        // Cosine distance
        var dot_ab: f64 = f64(0.0);
        var norm_a: f64 = f64(0.0);
        var norm_b: f64 = f64(0.0);
        for (var k = 0u; k < params.d; k = k + 1u) {
            let a_val = input_a[i * params.d + k];
            let b_val = input_b[j * params.d + k];
            dot_ab = dot_ab + a_val * b_val;
            norm_a = norm_a + a_val * a_val;
            norm_b = norm_b + b_val * b_val;
        }
        let cosine_sim = dot_ab / (sqrt(norm_a) * sqrt(norm_b) + f64(1e-30));
        dist = f64(1.0) - cosine_sim;
    }

    output[i * params.n + j] = dist;
}
