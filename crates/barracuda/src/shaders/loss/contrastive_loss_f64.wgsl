// SPDX-License-Identifier: AGPL-3.0-or-later
// Contrastive Loss (NT-Xent) - SimCLR, MoCo style (f64 canonical)
// Self-supervised learning by contrasting positive/negative pairs
//
// Algorithm:
// 1. Compute cosine similarity matrix: sim(i,j) = dot(z_i, z_j) / (||z_i|| * ||z_j||)
// 2. For each sample i, positive pair is at i+batch or i-batch
// 3. Loss = -log(exp(sim(i,pos)/tau) / sum(exp(sim(i,j)/tau) for j!=i))
//
// Used in: SimCLR, MoCo, self-supervised representation learning
// Benefits: Learns rich representations without labels

@group(0) @binding(0) var<storage, read> embeddings: array<f64>;  // [batch*2, embed_dim]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [batch_size] losses
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    batch_size: u32,
    embed_dim: u32,
    temperature: f64,
    _padding: u32,
}

// Compute cosine similarity between two embeddings
fn cosine_similarity(idx1: u32, idx2: u32, embed_dim: u32) -> f64 {
    var dot_product = 0.0;
    var norm1 = 0.0;
    var norm2 = 0.0;

    for (var d = 0u; d < embed_dim; d = d + 1u) {
        let emb1 = embeddings[idx1 * embed_dim + d];
        let emb2 = embeddings[idx2 * embed_dim + d];
        dot_product += emb1 * emb2;
        norm1 += emb1 * emb1;
        norm2 += emb2 * emb2;
    }

    let epsilon = 1e-8;
    return dot_product / (sqrt_f64(norm1) * sqrt_f64(norm2) + epsilon);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.batch_size {
        return;
    }

    let total_size = params.batch_size * 2u;

    // Positive pair index: first half pairs with second half
    let pos_idx = i + params.batch_size;

    // Compute similarity to positive (scaled by temperature)
    let pos_sim = cosine_similarity(i, pos_idx, params.embed_dim) / params.temperature;
    let pos_exp = exp_f64(pos_sim);

    // Compute denominator: sum of exp(sim(i,j)/tau) for all j != i
    var denom = 0.0;
    for (var j = 0u; j < total_size; j = j + 1u) {
        if j != i {
            let sim = cosine_similarity(i, j, params.embed_dim) / params.temperature;
            denom += exp_f64(sim);
        }
    }

    // NT-Xent loss: -log(exp(pos_sim) / denom)
    output[i] = -log_f64(pos_exp / denom);
}
