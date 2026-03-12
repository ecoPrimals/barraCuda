// SPDX-License-Identifier: AGPL-3.0-only
// multilabel_margin_loss_f64.wgsl - Multi-label Margin Loss (f64 canonical)
//
// Margin-based loss for multi-label classification
// Loss = sum_{y in Y, y' not in Y} max(0, 1 - (x[y] - x[y'])) / |Y||Y'|
//
// where Y = set of true labels

struct Params {
    batch_size: u32,
    num_classes: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [batch, num_classes] - scores
@group(0) @binding(1) var<storage, read> target_data: array<f64>;       // [batch, num_classes] - binary labels
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [batch] - per-sample loss
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    var loss_sum: f64 = 0.0;
    var num_pairs: u32 = 0u;

    // For each positive label
    for (var pos_c: u32 = 0u; pos_c < params.num_classes; pos_c = pos_c + 1u) {
        let is_positive = target_data[b * params.num_classes + pos_c] > 0.5;

        if (is_positive) {
            let pos_score = input[b * params.num_classes + pos_c];

            // Compare with negative labels
            for (var neg_c: u32 = 0u; neg_c < params.num_classes; neg_c = neg_c + 1u) {
                let is_negative = target_data[b * params.num_classes + neg_c] < 0.5;

                if (is_negative) {
                    let neg_score = input[b * params.num_classes + neg_c];
                    let margin_violation = 1.0 - (pos_score - neg_score);
                    loss_sum = loss_sum + max(0.0, margin_violation);
                    num_pairs = num_pairs + 1u;
                }
            }
        }
    }

    // Average over pairs
    output[b] = select(0.0, loss_sum / f64(num_pairs), num_pairs > 0u);
}
