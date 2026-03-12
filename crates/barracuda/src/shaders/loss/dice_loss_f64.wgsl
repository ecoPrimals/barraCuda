// SPDX-License-Identifier: AGPL-3.0-only
// dice_loss_f64.wgsl - Dice coefficient loss for segmentation (f64 canonical)
//
// Dice Loss = 1 - (2 * intersection + smooth_val) / (sum_pred + sum_target + smooth_val)
//
// Used extensively in medical image segmentation to handle class imbalance
// Range: [0, 1] where 0 = perfect overlap, 1 = no overlap

struct Params {
    size: u32,
    smooth_val: f64,         // Smoothing factor (typically 1.0) to avoid division by zero
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> predicted: array<f64>;  // Predicted probabilities [0, 1]
@group(0) @binding(1) var<storage, read> target_data: array<f64>;     // Ground truth [0, 1]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // Scalar loss value
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup shared memory for reduction
var<workgroup> shared_intersection: array<f64, 256>;
var<workgroup> shared_pred_sum: array<f64, 256>;
var<workgroup> shared_target_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Compute local contributions
    var local_intersection: f64 = 0.0;
    var local_pred_sum: f64 = 0.0;
    var local_target_sum: f64 = 0.0;

    if (idx < params.size) {
        let pred = predicted[idx];
        let targ = target_data[idx];

        local_intersection = pred * targ;
        local_pred_sum = pred;
        local_target_sum = targ;
    }

    // Store in shared memory
    shared_intersection[local_idx] = local_intersection;
    shared_pred_sum[local_idx] = local_pred_sum;
    shared_target_sum[local_idx] = local_target_sum;

    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_intersection[local_idx] = shared_intersection[local_idx] + shared_intersection[local_idx + stride];
            shared_pred_sum[local_idx] = shared_pred_sum[local_idx] + shared_pred_sum[local_idx + stride];
            shared_target_sum[local_idx] = shared_target_sum[local_idx] + shared_target_sum[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes result
    if (local_idx == 0u) {
        let intersection = shared_intersection[0];
        let pred_sum = shared_pred_sum[0];
        let target_sum = shared_target_sum[0];

        // Dice coefficient: (2 * intersection + smooth_val) / (pred_sum + target_sum + smooth_val)
        let dice = (2.0 * intersection + params.smooth_val) / (pred_sum + target_sum + params.smooth_val);

        // Dice loss = 1 - Dice coefficient
        let loss = 1.0 - dice;

        // Atomic add to output (in case of multiple workgroups)
        output[0] = loss;
    }
}
