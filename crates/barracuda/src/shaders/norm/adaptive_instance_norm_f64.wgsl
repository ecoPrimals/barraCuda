// SPDX-License-Identifier: AGPL-3.0-or-later
// Adaptive Instance Normalization (AdaIN) - Style transfer (f64 canonical)
//
// Transfers style from one image to another

struct Params {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    spatial_size: u32,
}

@group(0) @binding(0) var<storage, read> content: array<f64>;
@group(0) @binding(1) var<storage, read> style_mean: array<f64>;
@group(0) @binding(2) var<storage, read> style_std: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let total_elements = params.batch * params.channels * params.spatial_size;
    if (idx >= total_elements) {
        return;
    }

    // Compute batch and channel indices
    let spatial_idx = idx % params.spatial_size;
    let channel_idx = (idx / params.spatial_size) % params.channels;
    let batch_idx = idx / (params.channels * params.spatial_size);

    // Compute content statistics for this batch/channel
    // First pass: compute mean
    var content_mean: f64 = f64(0.0);
    for (var s = 0u; s < params.spatial_size; s = s + 1u) {
        let content_idx = batch_idx * params.channels * params.spatial_size +
                         channel_idx * params.spatial_size + s;
        content_mean = content_mean + content[content_idx];
    }
    content_mean = content_mean / f64(params.spatial_size);

    // Second pass: compute variance
    var content_var: f64 = f64(0.0);
    for (var s = 0u; s < params.spatial_size; s = s + 1u) {
        let content_idx = batch_idx * params.channels * params.spatial_size +
                         channel_idx * params.spatial_size + s;
        let diff = content[content_idx] - content_mean;
        content_var = content_var + diff * diff;
    }
    content_var = content_var / f64(params.spatial_size);
    let content_std = sqrt_f64(content_var);

    // Apply AdaIN: normalize content, then scale/shift to style
    let content_idx = batch_idx * params.channels * params.spatial_size +
                     channel_idx * params.spatial_size + spatial_idx;
    let normalized = (content[content_idx] - content_mean) / (content_std + f64(1e-5));
    output[content_idx] = normalized * style_std[channel_idx] + style_mean[channel_idx];
}
