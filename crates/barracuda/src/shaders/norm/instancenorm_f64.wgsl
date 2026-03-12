// SPDX-License-Identifier: AGPL-3.0-only
// Instance Normalization (f64 canonical)
// Normalizes each instance (sample) independently across spatial dimensions
//
// For input shape [batch, channels, height, width]:
// Computes mean and variance over (height, width) for each (batch, channel) pair
//
// InstanceNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
//
// Used in: Style transfer, GANs, real-time image generation
// Benefits: No dependency on batch, works well for style/texture tasks

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // Scale per channel
@group(0) @binding(2) var<storage, read> beta: array<f64>;   // Shift per channel
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    batch: u32,
    channels: u32,
    spatial_size: u32,  // height * width
    epsilon: f64,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_instances = params.batch * params.channels;
    
    if idx >= total_instances {
        return;
    }
    
    let b = idx / params.channels;
    let c = idx % params.channels;
    
    // Compute mean over spatial dimensions
    var sum: f64 = 0.0;
    let base_idx = b * (params.channels * params.spatial_size) + c * params.spatial_size;
    
    for (var i: u32 = 0u; i < params.spatial_size; i = i + 1u) {
        sum = sum + input[base_idx + i];
    }
    
    let mean = sum / f64(params.spatial_size);
    
    // Compute variance
    var var_sum: f64 = 0.0;
    for (var i: u32 = 0u; i < params.spatial_size; i = i + 1u) {
        let diff = input[base_idx + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    
    let variance = var_sum / f64(params.spatial_size);
    let std_dev = sqrt_f64(variance + params.epsilon);
    
    // Normalize and apply affine transform
    let g = gamma[c];
    let bt = beta[c];
    
    for (var i: u32 = 0u; i < params.spatial_size; i = i + 1u) {
        let normalized = (input[base_idx + i] - mean) / std_dev;
        output[base_idx + i] = g * normalized + bt;
    }
}
