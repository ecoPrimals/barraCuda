// SPDX-License-Identifier: AGPL-3.0-only
// Perceptual Loss - Feature-based perceptual loss (f64 canonical)
// Compares high-level features instead of pixels
// Used in style transfer and super-resolution
//
// Algorithm:
// If weights provided:
//   loss = sum(weights[i] * (features1[i] - features2[i])^2) / size
// Else:
//   loss = sum((features1[i] - features2[i])^2) / size
//
// Uses workgroup shared memory for correct float reduction

struct Params {
    size: u32,
    has_weights: u32, // 1 if weights provided, 0 otherwise
    num_weights: u32, // Number of weight groups
    num_partials: u32, // Number of workgroup partial results
}

@group(0) @binding(0) var<storage, read> features1: array<f64>;
@group(0) @binding(1) var<storage, read> features2: array<f64>;
@group(0) @binding(2) var<storage, read> weights: array<f64>;
@group(0) @binding(3) var<storage, read_write> loss_buffer: array<f64>;  // partial sums per workgroup
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_loss: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    var weighted_sq_diff: f64 = 0.0;
    if (idx < params.size) {
        let diff = features1[idx] - features2[idx];
        let sq_diff = diff * diff;
        if (params.has_weights == 1u) {
            let features_per_weight = params.size / params.num_weights;
            let weight_idx = idx / features_per_weight;
            weighted_sq_diff = weights[weight_idx] * sq_diff;
        } else {
            weighted_sq_diff = sq_diff;
        }
    }

    shared_loss[local_idx] = weighted_sq_diff;
    workgroupBarrier();

    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_loss[local_idx] = shared_loss[local_idx] + shared_loss[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        loss_buffer[workgroup_id.x] = shared_loss[0];
    }
}

// Second pass: sum partial results and compute mean
@compute @workgroup_size(256)
fn compute_mean_loss(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let num_wg = params.num_partials;

    var partial: f64 = 0.0;
    if (idx < num_wg) {
        partial = loss_buffer[idx];
    }
    shared_loss[local_idx] = partial;
    workgroupBarrier();

    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_loss[local_idx] = shared_loss[local_idx] + shared_loss[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        let total_loss = shared_loss[0];
        output[0] = total_loss / f64(params.size);
    }
}
