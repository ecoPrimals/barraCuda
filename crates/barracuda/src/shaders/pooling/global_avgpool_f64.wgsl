// SPDX-License-Identifier: AGPL-3.0-or-later
// Global Average Pooling
// Reduces spatial dimensions (H x W) to 1x1 by averaging across all spatial locations
//
// For input tensor [batch, channels, height, width]
// Output: [batch, channels, 1, 1]
//
// Used in: Modern CNNs (ResNet, EfficientNet) to replace fully connected layers
// Benefits: Reduces parameters, increases spatial invariance

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
    
    // Compute average across spatial dimensions (H x W)
    let spatial_size = params.height * params.width;
    var sum: f64 = 0.0;
    
    for (var h: u32 = 0u; h < params.height; h = h + 1u) {
        for (var w: u32 = 0u; w < params.width; w = w + 1u) {
            let input_idx = batch_idx * (params.channels * spatial_size) +
                           channel_idx * spatial_size +
                           h * params.width +
                           w;
            sum = sum + input[input_idx];
        }
    }
    
    output[idx] = sum / f64(spatial_size);
}
