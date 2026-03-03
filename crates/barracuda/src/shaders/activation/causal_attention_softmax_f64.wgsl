// Causal Attention Softmax: Apply softmax with causal masking (f64 canonical)
// Pass 2 of causal attention (reuses attention matmul and apply passes)
//
// **Causal Mask**: Position i can only attend to positions 0..=i (no future)
// mask[i,j] = -inf if j > i, else 0
//
// Input: attention scores [batch, heads, q_seq_len, kv_seq_len]
// Output: attention weights [batch, heads, q_seq_len, kv_seq_len] (with causal masking)
// For self-attention: q_seq_len == kv_seq_len

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

@group(0) @binding(0) var<storage, read> scores: array<f64>;
@group(0) @binding(1) var<storage, read_write> weights: array<f64>;
@group(0) @binding(2) var<uniform> params: AttentionParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.q_seq_len;

    if (idx >= total) {
        return;
    }

    // Decompose index
    let b = idx / (params.num_heads * params.q_seq_len);
    let h = (idx % (params.num_heads * params.q_seq_len)) / params.q_seq_len;
    let i = idx % params.q_seq_len; // query position

    let base_idx = b * params.num_heads * params.q_seq_len * params.kv_seq_len +
                   h * params.q_seq_len * params.kv_seq_len +
                   i * params.kv_seq_len;

    // Find max score (only over valid positions due to causal mask)
    var max_score = f64(-1e10);
    for (var j = 0u; j <= i; j++) {
        let score_idx = base_idx + j;
        max_score = max(max_score, scores[score_idx]);
    }

    // Compute exp and sum (with causal masking)
    var sum = f64(0.0);
    for (var j = 0u; j < params.kv_seq_len; j++) {
        let score_idx = base_idx + j;

        if (j <= i) {
            // Valid position: apply softmax
            let exp_val = exp_f64(scores[score_idx] - max_score);
            weights[score_idx] = exp_val;
            sum += exp_val;
        } else {
            // Future position: mask to zero (causal mask)
            weights[score_idx] = f64(0.0);
        }
    }

    // Normalize (only valid positions contribute to sum)
    for (var j = 0u; j <= i; j++) {
        let score_idx = base_idx + j;
        weights[score_idx] = weights[score_idx] / sum;
    }
}
