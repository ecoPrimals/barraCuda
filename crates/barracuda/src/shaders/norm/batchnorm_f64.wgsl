// SPDX-License-Identifier: AGPL-3.0-or-later
// BatchNorm: Batch normalization (f64 canonical)
// CUDA equivalent: cudnn::BatchNormalization
// Formula: output = (input - running_mean) / sqrt(running_var + epsilon) * gamma + beta
// Use cases: CNN normalization, accelerating training

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;
@group(0) @binding(2) var<storage, read> beta: array<f64>;
@group(0) @binding(3) var<storage, read> running_mean: array<f64>;
@group(0) @binding(4) var<storage, read> running_var: array<f64>;
@group(0) @binding(5) var<storage, read_write> output: array<f64>;

struct Params {
    batch_size: u32,
    channels: u32,
    spatial_size: u32,  // H * W
    epsilon: f64,
    training: u32,  // 0=inference, 1=training
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let total_size = params.batch_size * params.channels * params.spatial_size;

    if (gid >= total_size) {
        return;
    }

    // Calculate channel index
    let spatial_idx = gid % params.spatial_size;
    let channel_idx = (gid / params.spatial_size) % params.channels;

    // Normalize using running statistics (inference mode)
    let mean = running_mean[channel_idx];
    let variance = running_var[channel_idx];

    let normalized = (input[gid] - mean) / sqrt_f64(variance + params.epsilon);
    output[gid] = normalized * gamma[channel_idx] + beta[channel_idx];
}
