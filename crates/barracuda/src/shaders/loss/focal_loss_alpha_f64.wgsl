// SPDX-License-Identifier: AGPL-3.0-only
// focal_loss_alpha_f64.wgsl - Focal Loss with class weighting (f64 canonical)
//
// Extension of focal loss with per-class weights (alpha)
// Reference: "Focal Loss for Dense Object Detection" by Lin et al. (2017)
//
// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

struct Params {
    batch_size: u32,
    num_classes: u32,
    gamma: f64,      // Focusing parameter (typically 2.0)
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> predictions: array<f64>;  // [batch, num_classes] - probabilities
@group(0) @binding(1) var<storage, read> targets: array<u32>;      // [batch] - class indices
@group(0) @binding(2) var<storage, read> alpha: array<f64>;        // [num_classes] - class weights
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [batch] - per-sample loss
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    let target_class = targets[b];

    if (target_class >= params.num_classes) {
        output[b] = 0.0;
        return;
    }

    // Get predicted probability for true class
    let p_t = predictions[b * params.num_classes + target_class];
    let p_t_clamped = clamp(p_t, 1e-7, 1.0 - 1e-7);

    // Get class weight
    let alpha_t = alpha[target_class];

    // Focal loss: -α_t * (1 - p_t)^γ * log(p_t)
    let focal_weight = pow_f64(1.0 - p_t_clamped, params.gamma);
    let loss = -alpha_t * focal_weight * log_f64(p_t_clamped);

    output[b] = loss;
}
