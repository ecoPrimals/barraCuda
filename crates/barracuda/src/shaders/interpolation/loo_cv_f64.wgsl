// SPDX-License-Identifier: AGPL-3.0-or-later
// loo_cv_f64.wgsl — Leave-One-Out Cross-Validation for kernel methods (f64 canonical)
// LOO_i = (y_i - pred_i) / (1 - H_ii)
// H: hat matrix [N, N], y: targets [N], pred: predictions [N]
// H stored row-major, H[i,i] = H[i*N + i]

@group(0) @binding(0) var<storage, read> hat_matrix: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read> predictions: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    n: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    let n = params.n;
    let h_ii = hat_matrix[i * n + i];
    let denom = 1.0 - h_ii;

    // Avoid division by zero (H_ii = 1 means point has full influence - edge case)
    var loo_residual: f64;
    if (abs(denom) < 1e-10) {
        loo_residual = 0.0;
    } else {
        loo_residual = (y[i] - predictions[i]) / denom;
    }

    output[i] = loo_residual;
}
