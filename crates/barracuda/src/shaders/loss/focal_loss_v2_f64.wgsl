// SPDX-License-Identifier: AGPL-3.0-or-later
// Focal Loss v2 - Enhanced focal loss with alpha balancing (f64 canonical)
// Improved version with class balancing parameter
//
// Algorithm:
// For each element:
//   p_t = p if target=1, else (1-p)
//   focal_weight = alpha * (1 - p_t)^gamma if target=1, else (1-alpha) * p_t^gamma
//   bce = -(target * log(p) + (1-target) * log(1-p))
//   loss = focal_weight * bce
// Final: mean of all losses

struct Params {
    alpha: f64,
    gamma: f64,
    epsilon: f64,
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let pred = predictions[idx];
    let targ = targets[idx];

    // Clamp predictions for numerical stability
    let p = clamp(pred, params.epsilon, 1.0 - params.epsilon);

    // Compute p_t
    var p_t: f64;
    if (targ > 0.5) {
        p_t = p;
    } else {
        p_t = 1.0 - p;
    }

    // Compute focal weight
    var focal_weight: f64;
    if (targ > 0.5) {
        focal_weight = params.alpha * pow_f64(1.0 - p_t, params.gamma);
    } else {
        focal_weight = (1.0 - params.alpha) * pow_f64(p_t, params.gamma);
    }

    // Compute BCE
    let bce = -(targ * log_f64(p) + (1.0 - targ) * log_f64(1.0 - p));

    // Compute focal loss
    output[idx] = focal_weight * bce;
}
