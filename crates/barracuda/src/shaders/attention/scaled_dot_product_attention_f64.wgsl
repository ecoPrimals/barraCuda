// SPDX-License-Identifier: AGPL-3.0-or-later
// Scaled Dot-Product Attention - Transformer core operation (f64 canonical)
// attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//
// Reference: "Attention is All You Need" (Vaswani et al., 2017)
// Supports cross-attention: Q seq_len may differ from K/V seq_len.

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

@group(0) @binding(0) var<storage, read> query: array<f64>;        // [B, H, Sq, D]
@group(0) @binding(1) var<storage, read> key: array<f64>;          // [B, H, Skv, D]
@group(0) @binding(2) var<storage, read> value: array<f64>;        // [B, H, Skv, D]
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [B, H, Sq, D]
@group(0) @binding(4) var<uniform> params: AttentionParams;

@compute @workgroup_size(256)
fn compute_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let batch = global_id.x / (params.num_heads * params.q_seq_len);
    let head = (global_id.x / params.q_seq_len) % params.num_heads;
    let q_pos = global_id.x % params.q_seq_len;
    
    if (batch >= params.batch_size || head >= params.num_heads || q_pos >= params.q_seq_len) {
        return;
    }
    
    for (var k_pos = 0u; k_pos < params.kv_seq_len; k_pos = k_pos + 1u) {
        var score = f64(0.0);
        
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            let q_idx = batch * params.num_heads * params.q_seq_len * params.head_dim
                      + head * params.q_seq_len * params.head_dim
                      + q_pos * params.head_dim
                      + d;
            
            let k_idx = batch * params.num_heads * params.kv_seq_len * params.head_dim
                      + head * params.kv_seq_len * params.head_dim
                      + k_pos * params.head_dim
                      + d;
            
            score += query[q_idx] * key[k_idx];
        }
        
        let scale = sqrt_f64(f64(params.head_dim));
        score = score / scale;
    }
}
