// SPDX-License-Identifier: AGPL-3.0-only
// Global Max Pooling
// Reduces spatial dimensions (H x W) to 1x1 by taking maximum across all spatial locations
//
// For input tensor [batch, channels, height, width]
// Output: [batch, channels, 1, 1]
//
// Used in: CNNs for classification, attention mechanisms
// Benefits: Captures most salient features, reduces overfitting

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_outputs = params.batch_size * params.channels;
    
    if idx >= num_outputs {
        return;
    }
    
    let batch_idx = idx / params.channels;
    let channel_idx = idx % params.channels;
    
    // Compute maximum across spatial dimensions (H x W)
    let spatial_size = params.height * params.width;
    var max_val: f64 = -1e308;  // Very large negative number
    
    for (var h: u32 = 0u; h < params.height; h = h + 1u) {
        for (var w: u32 = 0u; w < params.width; w = w + 1u) {
            let input_idx = batch_idx * (params.channels * spatial_size) +
                           channel_idx * spatial_size +
                           h * params.width +
                           w;
            max_val = max(max_val, input[input_idx]);
        }
    }
    
    output[idx] = max_val;
}
