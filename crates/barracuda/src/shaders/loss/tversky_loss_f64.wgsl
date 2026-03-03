// Tversky Loss - Generalized Dice Loss with FP/FN weighting (f64 canonical)
// Allows control over false positive vs false negative trade-off
//
// TverskyIndex = TP / (TP + alpha*FP + beta*FN)
// TverskyLoss = 1 - TverskyIndex
//
// where:
// TP = true positives (intersection: pred * target)
// FP = false positives (pred_sum - intersection)
// FN = false negatives (target_sum - intersection)
// alpha = weight for false positives (typically 0.3-0.7)
// beta = weight for false negatives (typically 0.3-0.7)
//
// Special cases:
// - alpha = beta = 0.5: Equivalent to Dice Loss
// - alpha = beta = 1.0: Equivalent to Tanimoto coefficient
// - alpha < beta: Penalize false negatives more (recall-focused)
// - alpha > beta: Penalize false positives more (precision-focused)
//
// Used in: Medical image segmentation, imbalanced datasets, when FP/FN have different costs
// Benefits: Fine-grained control over precision-recall trade-off

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    alpha: f64,              // Weight for false positives (0.0-1.0, typically 0.3-0.7)
    beta: f64,               // Weight for false negatives (0.0-1.0, typically 0.3-0.7)
    smoothing: f64,          // Smoothing factor to avoid division by zero, typically 1.0
    batch_size: u32,
    elements_per_sample: u32,
}

// Shared memory for reduction within workgroup
var<workgroup> shared_intersection: array<f64, 256>;
var<workgroup> shared_pred_sum: array<f64, 256>;
var<workgroup> shared_target_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let batch_idx = workgroup_id.x;
    let local_idx = local_id.x;

    if batch_idx >= params.batch_size {
        return;
    }

    let base_idx = batch_idx * params.elements_per_sample;

    // Initialize shared memory
    shared_intersection[local_idx] = 0.0;
    shared_pred_sum[local_idx] = 0.0;
    shared_target_sum[local_idx] = 0.0;

    // Each thread processes multiple elements
    let threads_per_workgroup = 256u;
    let elements_per_thread = (params.elements_per_sample + threads_per_workgroup - 1u) / threads_per_workgroup;

    for (var i = 0u; i < elements_per_thread; i = i + 1u) {
        let idx = base_idx + local_idx + i * threads_per_workgroup;
        if idx < base_idx + params.elements_per_sample {
            let pred = predictions[idx];
            let targ = targets[idx];

            // Accumulate for Tversky calculation
            shared_intersection[local_idx] = shared_intersection[local_idx] + pred * targ;  // TP
            shared_pred_sum[local_idx] = shared_pred_sum[local_idx] + pred;                // TP + FP
            shared_target_sum[local_idx] = shared_target_sum[local_idx] + targ;            // TP + FN
        }
    }

    workgroupBarrier();

    // Reduction within workgroup
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if local_idx < stride {
            shared_intersection[local_idx] = shared_intersection[local_idx] + shared_intersection[local_idx + stride];
            shared_pred_sum[local_idx] = shared_pred_sum[local_idx] + shared_pred_sum[local_idx + stride];
            shared_target_sum[local_idx] = shared_target_sum[local_idx] + shared_target_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    // First thread computes Tversky loss for this sample
    if local_idx == 0u {
        let tp = shared_intersection[0];             // True positives
        let fp = shared_pred_sum[0] - tp;            // False positives
        let false_neg = shared_target_sum[0] - tp;   // False negatives

        // Tversky Index: TP / (TP + alpha*FP + beta*FN)
        let tversky_index = (tp + params.smoothing) /
                            (tp + params.alpha * fp + params.beta * false_neg + params.smoothing);

        // Tversky Loss = 1 - Tversky Index
        output[batch_idx] = 1.0 - tversky_index;
    }
}
