// SPDX-License-Identifier: AGPL-3.0-only
// Huber Loss (Smooth L1 Loss)
// Robust regression loss that's less sensitive to outliers than MSE
//
// Huber(x, y) = 0.5 * (x - y)²           if |x - y| <= delta
//              = delta * (|x - y| - 0.5 * delta)  otherwise
//
// Combines MSE (for small errors) with MAE (for large errors)
// Used in: Robust regression, reinforcement learning (DQN)

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    delta: f64,           // Threshold for switching from quadratic to linear
    reduction_mode: u32,  // 0=mean, 1=sum, 2=none
    size: u32,
    _padding: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }
    
    let pred = predictions[idx];
    let targ = targets[idx];
    let diff = abs(pred - targ);
    
    // Compute Huber loss
    var loss: f64;
    if diff <= params.delta {
        // Quadratic region (like MSE)
        loss = 0.5 * diff * diff;
    } else {
        // Linear region (like MAE)
        loss = params.delta * (diff - 0.5 * params.delta);
    }
    
    output[idx] = loss;
}
