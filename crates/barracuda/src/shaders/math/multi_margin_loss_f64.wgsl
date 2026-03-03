// multi_margin_loss_f64.wgsl - Multi-class Margin Loss (f64 canonical)
//
// Hinge loss for multi-class classification
// Loss = sum_{j≠y} max(0, margin - (x[y] - x[j]))^p / num_classes
//
// Used in SVM-style multi-class classification

struct Params {
    batch_size: u32,
    num_classes: u32,
    p: u32,          // 1 or 2 (power for the margin)
    margin: f64,     // Margin (typically 1.0)
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [batch, num_classes] - scores
@group(0) @binding(1) var<storage, read> target_data: array<u32>;       // [batch] - true class
@group(0) @binding(2) var<storage, read> weight: array<f64>;       // [num_classes] - class weights
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [batch] - per-sample loss
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    let true_class = target_data[b];

    if (true_class >= params.num_classes) {
        output[b] = 0.0;
        return;
    }

    let true_score = input[b * params.num_classes + true_class];

    var loss_sum: f64 = 0.0;

    // Sum over all classes except true class
    for (var c: u32 = 0u; c < params.num_classes; c = c + 1u) {
        if (c != true_class) {
            let class_score = input[b * params.num_classes + c];
            let margin_violation = params.margin - (true_score - class_score);
            let hinge = max(0.0, margin_violation);

            // Apply power
            let powered = select(hinge, hinge * hinge, params.p == 2u);

            // Apply class weight
            loss_sum = loss_sum + weight[c] * powered;
        }
    }

    // Average over classes
    output[b] = loss_sum / f64(params.num_classes);
}
