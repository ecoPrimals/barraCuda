// SPDX-License-Identifier: AGPL-3.0-or-later
// Covariance computation — f64 precision
// Cov(X,Y) = E[(X-μx)(Y-μy)] = Σ(x-μx)(y-μy) / (n - ddof)
//
// Input: two vectors x and y of same length (f64)
// Output: covariance value per pair
//
// Applications: portfolio theory, PCA, Kalman filters
// Reference: Standard statistical formula
//
// DF64 note: This shader uses only thread-local f64 accumulators — no
// var<workgroup> array<f64, N>. The automatic df64_rewrite is adequate
// for Hybrid devices; a hand-written DF64 variant is NOT needed here.

@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,        // Length of each vector
    num_pairs: u32,   // Number of (x,y) vector pairs
    stride: u32,      // Stride between vectors
    ddof: u32,        // Delta degrees of freedom (0 for population, 1 for sample)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_pairs) {
        return;
    }

    let size = params.size;
    let ddof = params.ddof;

    if (size <= ddof) {
        output[idx] = f64(0.0);
        return;
    }

    let x_offset = idx * params.stride;
    let y_offset = idx * params.stride;

    // Pass 1: compute means
    // @ilp_region begin — x and y accumulations are independent
    var sum_x: f64 = f64(0.0);
    var sum_y: f64 = f64(0.0);
    for (var i = 0u; i < size; i = i + 1u) {
        sum_x = sum_x + x[x_offset + i];
        sum_y = sum_y + y[y_offset + i];
    }
    let mean_x = sum_x / f64(size);
    let mean_y = sum_y / f64(size);
    // @ilp_region end

    // Pass 2: compute covariance
    var cov_sum: f64 = f64(0.0);
    for (var i = 0u; i < size; i = i + 1u) {
        let dx = x[x_offset + i] - mean_x;
        let dy = y[y_offset + i] - mean_y;
        cov_sum = cov_sum + dx * dy;
    }

    let result = cov_sum / f64(size - ddof);
    output[idx] = result;
}
