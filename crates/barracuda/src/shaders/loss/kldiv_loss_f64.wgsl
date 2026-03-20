// SPDX-License-Identifier: AGPL-3.0-or-later
// kldiv_loss_f64.wgsl - KL Divergence Loss (f64 canonical)
//
// Kullback-Leibler divergence between distributions
// KL(P||Q) = sum(P * log(P/Q))
//
// Note: Different from kl_divergence.wgsl which may have different semantics

struct Params {
    size: u32,
    log_target: u32,  // 1 if target is in log space, 0 otherwise
    reduction: u32,    // 0 = none, 1 = mean, 2 = sum, 3 = batchmean
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // Log probabilities (Q)
@group(0) @binding(1) var<storage, read> target_data: array<f64>;       // True probabilities (P)
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // KL divergence
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let log_q = input[idx];

    var p: f64;
    var log_p: f64;

    if (params.log_target == 1u) {
        log_p = target_data[idx];
        p = exp_f64(log_p);
    } else {
        p = target_data[idx];
        log_p = log_f64(p + 1e-12);
    }

    // KL(P||Q) = P * (log(P) - log(Q)) = P * log(P/Q)
    let kl = p * (log_p - log_q);

    output[idx] = kl;
}
