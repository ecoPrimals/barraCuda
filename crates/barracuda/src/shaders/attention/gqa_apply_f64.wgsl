// SPDX-License-Identifier: AGPL-3.0-or-later
// Grouped Query Attention Apply: Apply attention weights to values
// Pass 3 of GQA multi-pass attention implementation
//
// Computes: output[q_head, i, d] = sum_j (attention_weights[q_head, i, j] * V[kv_head, j, d])
// This is the final step: weighted sum of values
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

@group(0) @binding(0) var<storage, read> weights: array<f64>;   // [batch, num_q_heads, seq, seq]
@group(0) @binding(1) var<storage, read> value: array<f64>;     // [batch, num_kv_heads, seq, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [batch, num_q_heads, seq, head_dim]
@group(0) @binding(3) var<uniform> params: GQAParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_qhead = global_id.z;
    let i = global_id.y; // output position
    let d = global_id.x; // dimension
    
    if (i >= params.seq_len || d >= params.head_dim) {
        return;
    }
    
    let batch = batch_qhead / params.num_q_heads;
    let q_head = batch_qhead % params.num_q_heads;
    
    if (batch >= params.batch_size) {
        return;
    }
    
    // Map query head to corresponding KV head
    let kv_head = q_head / params.heads_per_group;
    
    // Compute weighted sum: sum_j (weights[q_head, i, j] * V[kv_head, j, d])
    var weighted_sum = 0.0;
    
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        // Weight indexing: [batch, num_q_heads, seq_len, seq_len]
        let weight_idx = batch * params.num_q_heads * params.seq_len * params.seq_len
                       + q_head * params.seq_len * params.seq_len
                       + i * params.seq_len
                       + j;
        
        // Value indexing: [batch, num_kv_heads, seq_len, head_dim]
        let value_idx = batch * params.num_kv_heads * params.seq_len * params.head_dim
                      + kv_head * params.seq_len * params.head_dim
                      + j * params.head_dim
                      + d;
        
        weighted_sum += weights[weight_idx] * value[value_idx];
    }
    
    // Store output: [batch, num_q_heads, seq_len, head_dim]
    let out_idx = batch * params.num_q_heads * params.seq_len * params.head_dim
                + q_head * params.seq_len * params.head_dim
                + i * params.head_dim
                + d;
    
    output[out_idx] = weighted_sum;
}
