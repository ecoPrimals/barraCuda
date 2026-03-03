// Focal Loss (f64 canonical)
// Addresses class imbalance in object detection by down-weighting easy examples
//
// FocalLoss = -alpha * (1 - p)^gamma * log(p)
// where p is the predicted probability for the correct class
//
// Parameters:
// - alpha: Balancing factor (typically 0.25)
// - gamma: Focusing parameter (typically 2.0)
//
// Used in: RetinaNet, object detection with severe class imbalance
// Benefits: Focuses training on hard examples, improves detection of rare classes

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    alpha: f64,           // Balancing factor, typically 0.25
    gamma: f64,           // Focusing parameter, typically 2.0
    epsilon: f64,         // Numerical stability
    reduction_mode: u32,  // 0=mean, 1=sum, 2=none
    size: u32,
    _pad1: vec3<u32>,     // 12 bytes
    _pad2: vec4<u32>,     // 16 bytes
    _pad3: vec4<u32>,     // 16 bytes
    _pad4: vec4<u32>,     // 16 bytes = Total 80 bytes
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }

    let pred = predictions[idx];
    let targ = targets[idx];

    // Clamp predictions for numerical stability
    let pred_clamped = clamp(pred, params.epsilon, 1.0 - params.epsilon);

    // Compute focal loss
    // For binary: FL = -alpha * (1 - p_t)^gamma * log(p_t)
    // where p_t = p if target=1, else (1-p)
    var p_t: f64;
    if targ > 0.5 {
        p_t = pred_clamped;
    } else {
        p_t = 1.0 - pred_clamped;
    }

    let focal_weight = pow_f64(1.0 - p_t, params.gamma);
    let loss = -params.alpha * focal_weight * log_f64(p_t);

    output[idx] = loss;
}
