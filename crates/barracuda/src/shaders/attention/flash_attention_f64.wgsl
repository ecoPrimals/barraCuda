// SPDX-License-Identifier: AGPL-3.0-only
// flash_attention_f64.wgsl - Memory-efficient attention mechanism (f64 canonical)
//
// Flash Attention: Memory-efficient attention that reduces memory usage from O(N²) to O(N)
// by computing attention in blocks and using tiling strategies.
//
// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
// by Dao et al. (2022)
//
// Simplified implementation for BarraCuda - Full version would require more advanced tiling

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
    scale: f64,
}

@group(0) @binding(0) var<storage, read> query: array<f64>;    // [seq_len, head_dim]
@group(0) @binding(1) var<storage, read> key: array<f64>;      // [seq_len, head_dim]
@group(0) @binding(2) var<storage, read> value: array<f64>;    // [seq_len, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [seq_len, head_dim]
@group(0) @binding(4) var<uniform> params: Params;

// Workgroup shared memory for tiling (reduces global memory access)
var<workgroup> tile_q: array<f64, 256>;
var<workgroup> tile_k: array<f64, 256>;
var<workgroup> tile_v: array<f64, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let seq_idx = global_id.x;
    let head_idx = global_id.y;
    
    if (seq_idx >= params.seq_len || head_idx >= params.head_dim) {
        return;
    }
    
    // Flash Attention Algorithm (Simplified):
    // 1. Compute attention scores for query position
    // 2. Apply softmax scaling
    // 3. Compute weighted sum with values
    
    var max_score: f64 = f64(-1e10);
    var sum_exp: f64 = f64(0.0);
    
    // First pass: Find max score and compute exp sum (numerically stable softmax)
    for (var k: u32 = 0u; k < params.seq_len; k = k + 1u) {
        var score: f64 = f64(0.0);
        
        // Compute dot product: Q[seq_idx] · K[k]
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            let q_idx = seq_idx * params.head_dim + d;
            let k_idx = k * params.head_dim + d;
            score = score + query[q_idx] * key[k_idx];
        }
        
        // Scale by sqrt(d_k) for stability
        score = score * params.scale;
        
        // Track maximum for numerical stability
        max_score = max(max_score, score);
    }
    
    // Second pass: Compute exp scores and sum
    var scores: array<f64, 256>; // Store attention scores (limited to 256 seq_len for now)
    for (var k: u32 = 0u; k < params.seq_len && k < 256u; k = k + 1u) {
        var score: f64 = f64(0.0);
        
        // Recompute dot product
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            let q_idx = seq_idx * params.head_dim + d;
            let k_idx = k * params.head_dim + d;
            score = score + query[q_idx] * key[k_idx];
        }
        
        score = score * params.scale;
        
        // Numerically stable exp
        let exp_score = exp_f64(score - max_score);
        scores[k] = exp_score;
        sum_exp = sum_exp + exp_score;
    }
    
    // Third pass: Compute output = softmax(scores) @ V
    for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
        var weighted_sum: f64 = f64(0.0);
        
        for (var k: u32 = 0u; k < params.seq_len && k < 256u; k = k + 1u) {
            let attention_weight = scores[k] / sum_exp;
            let v_idx = k * params.head_dim + d;
            weighted_sum = weighted_sum + attention_weight * value[v_idx];
        }
        
        let out_idx = seq_idx * params.head_dim + d;
        output[out_idx] = weighted_sum;
    }
}
