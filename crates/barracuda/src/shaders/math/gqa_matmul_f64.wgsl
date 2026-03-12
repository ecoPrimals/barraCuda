// SPDX-License-Identifier: AGPL-3.0-only
// Grouped Query Attention Matrix Multiplication: Compute QK^T scores (f64 canonical)
// Pass 1 of GQA multi-pass attention implementation
//
// Computes: scores[i,j] = Q[q_head, i] · K[kv_head, j] / sqrt(d_k)
// Where Q is [batch, num_q_heads, seq_len, head_dim]
//       K is [batch, num_kv_heads, seq_len, head_dim]
//       scores is [batch, num_q_heads, seq_len, seq_len]
//
// Key difference: Multiple query heads share the same KV head
// kv_head = q_head / heads_per_group

struct GQAParams {
    batch_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    heads_per_group: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f64>;     // [batch, num_q_heads, seq, head_dim]
@group(0) @binding(1) var<storage, read> key: array<f64>;       // [batch, num_kv_heads, seq, head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f64>; // [batch, num_q_heads, seq, seq]
@group(0) @binding(3) var<uniform> params: GQAParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_qhead = global_id.z;
    let i = global_id.y; // query position
    let j = global_id.x; // key position

    if (i >= params.seq_len || j >= params.seq_len) {
        return;
    }

    let batch = batch_qhead / params.num_q_heads;
    let q_head = batch_qhead % params.num_q_heads;

    if (batch >= params.batch_size) {
        return;
    }

    // Map query head to corresponding KV head
    // Multiple query heads share the same KV head
    let kv_head = q_head / params.heads_per_group;

    // Compute dot product: Q[q_head, i] · K[kv_head, j]
    var score = 0.0;
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        // Query indexing: [batch, num_q_heads, seq_len, head_dim]
        let q_idx = batch * params.num_q_heads * params.seq_len * params.head_dim
                  + q_head * params.seq_len * params.head_dim
                  + i * params.head_dim
                  + d;

        // Key indexing: [batch, num_kv_heads, seq_len, head_dim]
        let k_idx = batch * params.num_kv_heads * params.seq_len * params.head_dim
                  + kv_head * params.seq_len * params.head_dim
                  + j * params.head_dim
                  + d;

        score += query[q_idx] * key[k_idx];
    }

    // Scale by sqrt(d_k) for numerical stability
    let scale = sqrt_f64(f64(params.head_dim));
    score = score / scale;

    // Store score: [batch, num_q_heads, seq_len, seq_len]
    let score_idx = batch * params.num_q_heads * params.seq_len * params.seq_len
                  + q_head * params.seq_len * params.seq_len
                  + i * params.seq_len
                  + j;

    scores[score_idx] = score;
}
