// Smooth L1 Loss (Huber Loss variant) - WGSL Shader
//
// Deep Debt: Universal compute for all chips!
// Pattern: Element-wise loss computation
//
// Algorithm:
//   diff = |pred - target|
//   if diff < beta:
//       loss = 0.5 * diff² / beta
//   else:
//       loss = diff - 0.5 * beta
//
// Used by: Object detection (Faster R-CNN), robust regression

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    beta: f64,
    size: u32,
    _padding: vec2<u32>,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let pred = predictions[idx];
    let targ = targets[idx];
    let diff = abs(pred - targ);
    
    // Smooth L1: quadratic below beta, linear above beta
    var loss: f64;
    if (diff < params.beta) {
        // Quadratic region: 0.5 * diff² / beta
        loss = 0.5 * diff * diff / params.beta;
    } else {
        // Linear region: diff - 0.5 * beta
        loss = diff - 0.5 * params.beta;
    }
    
    output[idx] = loss;
}
