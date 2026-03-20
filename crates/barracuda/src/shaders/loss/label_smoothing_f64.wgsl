// SPDX-License-Identifier: AGPL-3.0-or-later
// label_smoothing_f64.wgsl - Label Smoothing for classification (f64 canonical)
//
// Prevents overconfidence by smoothing hard labels
// Reference: "Rethinking the Inception Architecture" by Szegedy et al. (2016)
//
// Smoothed label: y_smooth = (1 - ε) * y + ε / num_classes

struct Params {
    batch_size: u32,
    num_classes: u32,
    smoothing: f64,  // ε, typically 0.1
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> labels: array<u32>;       // [batch_size] - class indices
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // [batch_size, num_classes]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    let true_class = labels[b];
    let confidence = 1.0 - params.smoothing;
    let other_prob = params.smoothing / f64(params.num_classes);

    for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) {
        let idx = b * params.num_classes + c;
        output[idx] = select(other_prob, confidence + other_prob, c == true_class);
    }
}
