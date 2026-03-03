// Lovasz-Softmax Loss (f64 canonical)
// IoU-optimized loss for semantic segmentation
//
// The Lovasz loss is a convex surrogate for the IoU (Jaccard) loss.
// It directly optimizes the Intersection over Union (IoU) metric.
//
// Steps:
// 1. Sort errors (1 - p_true) in descending order
// 2. Compute cumulative intersection
// 3. Apply Lovasz extension to IoU
//
// Used in: Semantic segmentation (better than cross-entropy for IoU)
// Reference: "The Lovász-Softmax loss" (Berman et al., CVPR 2018)

@group(0) @binding(0) var<storage, read> predictions: array<f64>;  // Probabilities [0, 1]
@group(0) @binding(1) var<storage, read> targets: array<f64>;      // Ground truth [0, 1]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // Per-element loss
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,
    smoothing: f64,  // Smoothing factor for numerical stability
    _padding: vec2<u32>,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }

    let pred = predictions[idx];
    let targ = targets[idx];

    // Compute error: (1 - p) where p is probability of true class
    let error = max(0.0, 1.0 - pred * targ + params.smoothing);

    // Store error for Lovasz extension
    // Note: Full Lovasz requires sorting, which is complex in compute shaders
    // This simplified version computes element-wise error that can be
    // reduced externally for full Lovasz computation
    output[idx] = error;
}
