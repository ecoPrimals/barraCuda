// SPDX-License-Identifier: AGPL-3.0-or-later
// MAE (Mean Absolute Error) Loss / L1 Loss
// Loss function less sensitive to outliers than MSE
//
// Formula: MAE = mean(|predictions - targets|)
//
// Properties:
// - More robust to outliers than MSE
// - Linear gradients (unlike quadratic for MSE)
// - Used in regression when outliers expected

@group(0) @binding(0) var<storage, read> predictions: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    reduction_mode: u32,  // 0=mean, 1=sum, 2=none
    size: u32,            // Total number of elements
    _padding: vec2<u32>,  // Alignment to 16 bytes
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }
    
    let pred = predictions[idx];
    let targ = targets[idx];
    
    // Compute absolute error
    let abs_error = abs(pred - targ);
    
    // Apply reduction based on mode
    if params.reduction_mode == 2u {  // None: per-element
        output[idx] = abs_error;
    } else {
        // For mean/sum, we write to output and reduce later
        output[idx] = abs_error;
    }
}
