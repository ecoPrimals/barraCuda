// SPDX-License-Identifier: AGPL-3.0-or-later
// L1 Loss - Mean Absolute Error
// loss = mean(|predictions - targets|)

struct Params {
    size: u32,
    reduction: u32,  // 0=none, 1=mean, 2=sum
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
    let target_data = targets[idx];
    let abs_diff = abs(pred - target_data);
    
    // Store individual losses
    output[idx] = abs_diff;
}

// Note: Reduction (mean/sum) would require a second pass or atomic operations
// For now, we compute element-wise losses and let the CPU handle reduction
