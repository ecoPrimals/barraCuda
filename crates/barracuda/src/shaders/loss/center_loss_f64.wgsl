// center_loss_f64.wgsl - Center Loss for metric learning (f64 canonical)
//
// Learns class centers and penalizes intra-class variance
// Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition" by Wen et al. (2016)
//
// Loss = (1/2) * ||features - centers[label]||²

struct Params {
    batch_size: u32,
    feature_dim: u32,
    num_classes: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> features: array<f64>;     // [batch, feature_dim]
@group(0) @binding(1) var<storage, read> centers: array<f64>;      // [num_classes, feature_dim]
@group(0) @binding(2) var<storage, read> labels: array<u32>;       // [batch]
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [batch] - per-sample loss
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    let label = labels[b];

    if (label >= params.num_classes) {
        output[b] = 0.0;
        return;
    }

    // Compute ||features - center||²
    var dist_sq: f64 = 0.0;

    for (var f: u32 = 0u; f < params.feature_dim; f = f + 1u) {
        let feat_idx = b * params.feature_dim + f;
        let center_idx = label * params.feature_dim + f;

        let diff = features[feat_idx] - centers[center_idx];
        dist_sq = dist_sq + diff * diff;
    }

    // Loss = (1/2) * distance²
    output[b] = 0.5 * dist_sq;
}
