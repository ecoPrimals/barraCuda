// SPDX-License-Identifier: AGPL-3.0-or-later
// Attention Matrix Multiplication: Compute QK^T scores (f64 canonical)
// Pass 1 of multi-pass attention implementation
//
// Computes: scores[i,j] = Q[i] · K[j] / sqrt(d_k)
// Q: [batch, heads, q_seq_len, head_dim]
// K: [batch, heads, kv_seq_len, head_dim]
// scores: [batch, heads, q_seq_len, kv_seq_len]

struct AttentionParams {
    batch_size: u32,
    num_heads: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f64>;
@group(0) @binding(1) var<storage, read> key: array<f64>;
@group(0) @binding(2) var<storage, read_write> scores: array<f64>;
@group(0) @binding(3) var<uniform> params: AttentionParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_head = global_id.z;
    let i = global_id.y; // query position
    let j = global_id.x; // key position
    
    if (i >= params.q_seq_len || j >= params.kv_seq_len) {
        return;
    }
    
    let batch = batch_head / params.num_heads;
    let head = batch_head % params.num_heads;
    
    if (batch >= params.batch_size) {
        return;
    }
    
    var score = f64(0.0);
    for (var d = 0u; d < params.head_dim; d = d + 1u) {
        let q_idx = batch * params.num_heads * params.q_seq_len * params.head_dim
                  + head * params.q_seq_len * params.head_dim
                  + i * params.head_dim
                  + d;
        
        let k_idx = batch * params.num_heads * params.kv_seq_len * params.head_dim
                  + head * params.kv_seq_len * params.head_dim
                  + j * params.head_dim
                  + d;
        
        score += query[q_idx] * key[k_idx];
    }
    
    let scale = sqrt_f64(f64(params.head_dim));
    score = score / scale;
    
    let score_idx = batch * params.num_heads * params.q_seq_len * params.kv_seq_len
                  + head * params.q_seq_len * params.kv_seq_len
                  + i * params.kv_seq_len
                  + j;
    
    scores[score_idx] = score;
}
