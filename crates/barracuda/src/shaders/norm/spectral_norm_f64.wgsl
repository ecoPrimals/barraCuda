// SPDX-License-Identifier: AGPL-3.0-only
// spectral_norm.wgsl - Spectral Normalization (f64 canonical)
//
// Normalizes weights by their spectral norm (largest singular value)
// Reference: "Spectral Normalization for GANs" by Miyato et al. (2018)
//
// Used in GANs to stabilize training by constraining Lipschitz constant

struct Params {
    rows: u32,
    cols: u32,
    num_iterations: u32,  // Power iteration steps (typically 1)
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> weight: array<f64>;  // [rows, cols]
@group(0) @binding(1) var<storage, read_write> u: array<f64>;       // [rows] - left singular vector
@group(0) @binding(2) var<storage, read_write> v: array<f64>;       // [cols] - right singular vector
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Power iteration to estimate largest singular value
    for (var iter: u32 = 0u; iter < params.num_iterations; iter = iter + 1u) {
        // v = W^T * u / ||W^T * u||
        if (idx < params.cols) {
            var sum: f64 = 0.0;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                sum = sum + weight[r * params.cols + idx] * u[r];
            }
            v[idx] = sum;
        }

        workgroupBarrier();

        // Normalize v
        if (idx == 0u) {
            var norm: f64 = 0.0;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                norm = norm + v[c] * v[c];
            }
            norm = sqrt_f64(norm + 1e-12);

            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                v[c] = v[c] / norm;
            }
        }

        workgroupBarrier();

        // u = W * v / ||W * v||
        if (idx < params.rows) {
            var sum: f64 = 0.0;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                sum = sum + weight[idx * params.cols + c] * v[c];
            }
            u[idx] = sum;
        }

        workgroupBarrier();

        // Normalize u
        if (idx == 0u) {
            var norm: f64 = 0.0;
            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                norm = norm + u[r] * u[r];
            }
            norm = sqrt_f64(norm + 1e-12);

            for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
                u[r] = u[r] / norm;
            }
        }

        workgroupBarrier();
    }

    // Compute spectral norm: σ = u^T * W * v
    if (idx == 0u) {
        var sigma: f64 = 0.0;
        for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
            var row_sum: f64 = 0.0;
            for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
                row_sum = row_sum + weight[r * params.cols + c] * v[c];
            }
            sigma = sigma + u[r] * row_sum;
        }

        // Normalize weights by spectral norm
        let scale = 1.0 / (sigma + 1e-12);
        for (var i: u32 = 0u; i < params.rows * params.cols; i = i + 1u) {
            weight[i] = weight[i] * scale;
        }
    }
}
