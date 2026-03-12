// SPDX-License-Identifier: AGPL-3.0-only
// Pearson correlation coefficient for batch vectors — f64 precision
// r = Σ(x-μx)(y-μy) / (σx·σy·n)
//
// Input: two vectors x and y of same length (f64)
// Output: single correlation coefficient per pair
//
// Applications: signal correlation, feature selection, portfolio analysis
// Reference: Standard statistical formula

@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,        // Length of each vector
    num_pairs: u32,   // Number of (x,y) vector pairs
    stride: u32,      // Stride between vectors (typically == size)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_pairs) {
        return;
    }

    let x_offset = idx * params.stride;
    let y_offset = idx * params.stride;
    let size = params.size;

    // Compute means
    var sum_x: f64 = f64(0.0);
    var sum_y: f64 = f64(0.0);
    for (var i = 0u; i < size; i = i + 1u) {
        sum_x = sum_x + x[x_offset + i];
        sum_y = sum_y + y[y_offset + i];
    }
    let mean_x = sum_x / f64(size);
    let mean_y = sum_y / f64(size);

    // Compute covariance and standard deviations
    var cov: f64 = f64(0.0);
    var var_x: f64 = f64(0.0);
    var var_y: f64 = f64(0.0);

    for (var i = 0u; i < size; i = i + 1u) {
        let dx = x[x_offset + i] - mean_x;
        let dy = y[y_offset + i] - mean_y;
        cov = cov + dx * dy;
        var_x = var_x + dx * dx;
        var_y = var_y + dy * dy;
    }

    let denom = sqrt(var_x * var_y);
    if (denom < f64(1e-15)) {
        output[idx] = f64(0.0);  // Avoid division by zero
        return;
    }

    let result = cov / denom;
    output[idx] = result;
}
