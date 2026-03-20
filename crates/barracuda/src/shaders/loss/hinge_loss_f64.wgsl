// SPDX-License-Identifier: AGPL-3.0-or-later
// Hinge Loss - SVM-style classification loss
// Used for support vector machines and max-margin classification
//
// HingeLoss = max(0, 1 - y * pred)
// where y ∈ {-1, +1} is true label, pred is prediction
//
// Key properties:
// - Zero loss when prediction has correct sign and magnitude > 1
// - Linear penalty for incorrect predictions
// - Encourages max-margin separation
//
// Used in: SVMs, max-margin classifiers

@group(0) @binding(0) var<storage, read> predictions: array<f64>;
@group(0) @binding(1) var<storage, read> targets: array<f64>;  // Should be -1 or +1
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,
    margin: f64,  // Typically 1.0 for standard hinge loss
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
    
    // Hinge loss: max(0, margin - targ * pred)
    // For standard SVM: margin = 1.0
    let loss = max(0.0, params.margin - targ * pred);
    
    output[idx] = loss;
}
