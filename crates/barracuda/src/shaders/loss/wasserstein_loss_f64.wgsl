// SPDX-License-Identifier: AGPL-3.0-or-later
// wasserstein_loss_f64.wgsl - Wasserstein Loss (1D case) (f64 canonical)
//
// Wasserstein distance for 1D distributions (Earth Mover's Distance)
// Efficient computation using cumulative distribution functions
//
// W_1(P, Q) = ∫|CDF_P(x) - CDF_Q(x)| dx

struct Params {
    size: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0) var<storage, read> pred: array<f64>;         // Predicted distribution
@group(0) @binding(1) var<storage, read> target_data: array<f64>;       // Target distribution
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [1] - distance
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Compute cumulative sums (CDFs)
    var cdf_pred: f64 = 0.0;
    var cdf_target: f64 = 0.0;

    for (var i: u32 = 0u; i <= idx; i = i + 1u) {
        cdf_pred = cdf_pred + pred[i];
        cdf_target = cdf_target + target_data[i];
    }

    // Store difference
    output[idx] = abs(cdf_pred - cdf_target);
}
