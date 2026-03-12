// SPDX-License-Identifier: AGPL-3.0-only
// Triplet Loss - Learn embeddings where similar items are close, dissimilar items are far (f64 canonical)
//
// For each triplet (anchor, positive, negative):
// - anchor: reference sample
// - positive: similar sample (same class/identity)
// - negative: dissimilar sample (different class/identity)
//
// Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
//
// where:
// - d(a, b) = distance metric (typically L2 Euclidean distance)
// - margin = minimum separation between positive and negative distances
//
// Goal: Pull positives closer, push negatives farther
// - d(anchor, positive) should be small
// - d(anchor, negative) should be large
// - Difference should be at least 'margin'
//
// Used in: Face recognition, person re-identification, metric learning, similarity search
// Benefits: Learns discriminative embeddings, no explicit classification layer needed

@group(0) @binding(0) var<storage, read> anchors: array<f64>;       // [batch, embedding_dim]
@group(0) @binding(1) var<storage, read> positives: array<f64>;     // [batch, embedding_dim]
@group(0) @binding(2) var<storage, read> negatives: array<f64>;     // [batch, embedding_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f64>;  // [batch] - loss per sample
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    batch_size: u32,
    embedding_dim: u32,
    margin: f64,          // Margin for triplet separation (typically 0.2-1.0)
    distance_type: u32,   // 0=L2 Euclidean, 1=cosine distance
}

// Compute L2 (Euclidean) distance squared between two embeddings
fn l2_distance_squared(idx: u32, dim: u32) -> f64 {
    var sum = 0.0;
    let base_idx = idx * dim;

    for (var i = 0u; i < dim; i = i + 1u) {
        let a = anchors[base_idx + i];
        let p = positives[base_idx + i];
        let diff = a - p;
        sum = sum + diff * diff;
    }

    return sum;
}

// Compute L2 distance between anchor and negative
fn l2_distance_squared_neg(idx: u32, dim: u32) -> f64 {
    var sum = 0.0;
    let base_idx = idx * dim;

    for (var i = 0u; i < dim; i = i + 1u) {
        let a = anchors[base_idx + i];
        let n = negatives[base_idx + i];
        let diff = a - n;
        sum = sum + diff * diff;
    }

    return sum;
}

// Compute cosine distance (1 - cosine similarity)
fn cosine_distance(idx: u32, dim: u32, is_negative: bool) -> f64 {
    var dot_product = 0.0;
    var norm_a = 0.0;
    var norm_other = 0.0;
    let base_idx = idx * dim;

    for (var i = 0u; i < dim; i = i + 1u) {
        let a = anchors[base_idx + i];
        let other = select(positives[base_idx + i], negatives[base_idx + i], is_negative);

        dot_product = dot_product + a * other;
        norm_a = norm_a + a * a;
        norm_other = norm_other + other * other;
    }

    // Cosine similarity = dot / (||a|| * ||b||)
    let cosine_sim = dot_product / (sqrt_f64(norm_a) * sqrt_f64(norm_other) + 1e-8);

    // Cosine distance = 1 - cosine_similarity
    return 1.0 - cosine_sim;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if idx >= params.batch_size {
        return;
    }

    var dist_pos: f64;
    var dist_neg: f64;

    if params.distance_type == 0u {
        // L2 Euclidean distance (squared for efficiency)
        dist_pos = l2_distance_squared(idx, params.embedding_dim);
        dist_neg = l2_distance_squared_neg(idx, params.embedding_dim);
    } else {
        // Cosine distance
        dist_pos = cosine_distance(idx, params.embedding_dim, false);
        dist_neg = cosine_distance(idx, params.embedding_dim, true);
    }

    // Triplet loss: max(0, d(a,p) - d(a,n) + margin)
    // If dist_pos < dist_neg by at least margin, loss = 0 (good triplet)
    // Otherwise, loss > 0 (need to pull positive closer or push negative farther)
    let loss = max(0.0, dist_pos - dist_neg + params.margin);

    output[idx] = loss;
}
