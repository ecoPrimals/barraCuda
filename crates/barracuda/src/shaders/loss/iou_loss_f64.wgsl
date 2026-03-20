// SPDX-License-Identifier: AGPL-3.0-or-later
// IoU Loss - Intersection over Union loss (f64 canonical)
// Direct optimization of IoU metric
// Used in segmentation and object detection
//
// Algorithm:
// IoU = (intersection + smooth_val) / (union + smooth_val)
// IoU Loss = 1 - IoU
// where intersection = sum(predictions * targets)
//       union = sum(predictions + targets - predictions * targets)
//
// Uses workgroup shared memory for correct float reduction (not broken atomic bitcast)

struct Params {
    size: u32,
    smooth_val: f64,
    num_partials: u32,  // number of workgroup partial results to reduce in pass 2
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> intersection_buffer: array<f64>;  // partial sums per workgroup
@group(0) @binding(3) var<storage, read_write> union_buffer: array<f64>;         // partial sums per workgroup
@group(0) @binding(4) var<storage, read_write> output: array<f64>;              // [1] - final loss
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_intersection: array<f64, 256>;
var<workgroup> shared_union: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    var local_intersection: f64 = 0.0;
    var local_union: f64 = 0.0;

    if (idx < params.size) {
        let pred = predictions[idx];
        let targ = targets[idx];
        local_intersection = pred * targ;
        local_union = pred + targ - local_intersection;
    }

    shared_intersection[local_idx] = local_intersection;
    shared_union[local_idx] = local_union;

    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_intersection[local_idx] = shared_intersection[local_idx] + shared_intersection[local_idx + stride];
            shared_union[local_idx] = shared_union[local_idx] + shared_union[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // First thread writes workgroup partial result
    if (local_idx == 0u) {
        intersection_buffer[workgroup_id.x] = shared_intersection[0];
        union_buffer[workgroup_id.x] = shared_union[0];
    }
}

// Second pass: sum partial results and compute final loss (single workgroup)
@compute @workgroup_size(256)
fn compute_loss(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let num_wg = params.num_partials;

    // Load partial results into shared memory
    var partial_int: f64 = 0.0;
    var partial_un: f64 = 0.0;
    if (idx < num_wg) {
        partial_int = intersection_buffer[idx];
        partial_un = union_buffer[idx];
    }
    shared_intersection[local_idx] = partial_int;
    shared_union[local_idx] = partial_un;

    workgroupBarrier();

    // Reduce
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_intersection[local_idx] = shared_intersection[local_idx] + shared_intersection[local_idx + stride];
            shared_union[local_idx] = shared_union[local_idx] + shared_union[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        let intersection = shared_intersection[0];
        let union_val = shared_union[0];
        let iou = (intersection + params.smooth_val) / (union_val + params.smooth_val + 1e-8);
        output[0] = 1.0 - iou;
    }
}
