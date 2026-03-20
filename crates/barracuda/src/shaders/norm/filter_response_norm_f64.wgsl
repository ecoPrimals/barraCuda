// SPDX-License-Identifier: AGPL-3.0-or-later
// Filter Response Normalization (FRN) - Normalization without batch dependency (f64 canonical)
// Normalizes activations per filter, not per batch
// Enables single-sample inference
//
// Algorithm:
// 1. Compute squared norm for each filter: nu = sqrt(sum(x^2) / spatial_size)
// 2. Normalize: x_norm = x / (nu + epsilon)
// 3. Scale and shift: output = gamma * x_norm + beta

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    spatial_size: u32,
    epsilon: f64,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // [channels]
@group(0) @binding(2) var<storage, read> beta: array<f64>;     // [channels]
@group(0) @binding(3) var<storage, read_write> sum_sq_buffer: array<f64>; // [batch * channels]
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_sq: array<f64, 256>;

// Step 1: Compute squared sum per filter via workgroup reduction
// One workgroup per (batch, channel)
@compute @workgroup_size(256)
fn compute_sum_sq(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let filter_idx = workgroup_id.x; // batch * channels + channel
    let base = filter_idx * params.spatial_size;

    var sq_val = f64(0.0);
    if (tid < params.spatial_size) {
        let val = input[base + tid];
        sq_val = val * val;
    }
    shared_sq[tid] = sq_val;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        sum_sq_buffer[filter_idx] = shared_sq[0];
    }
}

// Step 2: Normalize and apply scale/shift
@compute @workgroup_size(256)
fn normalize_and_scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.channels * params.spatial_size;

    if (idx >= total_elements) {
        return;
    }

    let channel_idx = (idx / params.spatial_size) % params.channels;
    let batch_idx = idx / (params.channels * params.spatial_size);
    let buffer_idx = batch_idx * params.channels + channel_idx;

    let sum_sq = sum_sq_buffer[buffer_idx];
    let nu = sqrt_f64(sum_sq / f64(params.spatial_size));

    // Normalize: x_norm = x / (nu + epsilon), ensure denominator is never zero
    let denom = nu + params.epsilon;
    let normalized = input[idx] / max(denom, f64(1e-8));

    output[idx] = gamma[channel_idx] * normalized + beta[channel_idx];
}
