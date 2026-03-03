// Attention Softmax: Apply softmax to attention scores (f64 canonical)
// Pass 2 of multi-pass attention implementation
//
// Computes: weights[i,j] = exp(scores[i,j] - max) / sum(exp(scores[i,:] - max))
// Applied row-wise (per query position). Each row has kv_seq_len entries.
// scores/weights: [batch, heads, q_seq_len, kv_seq_len]

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

    let batch = idx / (params.num_heads * params.q_seq_len);
    let head = (idx / params.q_seq_len) % params.num_heads;
    let i = idx % params.q_seq_len;

    let base_idx = batch * params.num_heads * params.q_seq_len * params.kv_seq_len
                 + head * params.q_seq_len * params.kv_seq_len
                 + i * params.kv_seq_len;

    var max_score = f64(-1e9);
    for (var j = 0u; j < params.kv_seq_len; j = j + 1u) {
        let score = scores[base_idx + j];
        max_score = max(max_score, score);
    }

    var sum = f64(0.0);
    for (var j = 0u; j < params.kv_seq_len; j = j + 1u) {
        let score = scores[base_idx + j];
        let exp_score = exp_f64(score - max_score);
        weights[base_idx + j] = exp_score;
        sum += exp_score;
    }

    for (var j = 0u; j < params.kv_seq_len; j = j + 1u) {
        weights[base_idx + j] = weights[base_idx + j] / sum;
    }
}
