// SPDX-License-Identifier: AGPL-3.0-only
// Grouped Query Attention Softmax: Apply softmax to attention scores (f64 canonical)
// Pass 2 of GQA multi-pass attention implementation
//
// Computes: attention_weights[i,j] = exp(scores[i,j] - max) / sum(exp(scores[i,:] - max))
// Applied row-wise (per query position)
//
// Note: Scores are [batch, num_q_heads, seq_len, seq_len]
// Each query head has its own attention weights

struct GQAParams {
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    heads_per_group: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f64>;    // [batch, num_q_heads, seq, seq]
@group(0) @binding(1) var<storage, read_write> weights: array<f64>; // [batch, num_q_heads, seq, seq]
@group(0) @binding(2) var<uniform> params: GQAParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_q_heads * params.seq_len;

    if (idx >= total) {
        return;
    }

    let batch = idx / (params.num_q_heads * params.seq_len);
    let q_head = (idx / params.seq_len) % params.num_q_heads;
    let i = idx % params.seq_len;

    let base_idx = batch * params.num_q_heads * params.seq_len * params.seq_len
                 + q_head * params.seq_len * params.seq_len
                 + i * params.seq_len;

    // Find max for numerical stability
    var max_score = f64(-1e9);
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        let score = scores[base_idx + j];
        max_score = max(max_score, score);
    }

    // Compute exp and sum
    var sum = f64(0.0);
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        let score = scores[base_idx + j];
        let exp_score = exp_f64(score - max_score);
        weights[base_idx + j] = exp_score;
        sum += exp_score;
    }

    // Normalize
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        weights[base_idx + j] = weights[base_idx + j] / sum;
    }
}
