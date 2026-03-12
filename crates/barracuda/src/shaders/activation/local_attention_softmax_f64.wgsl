// SPDX-License-Identifier: AGPL-3.0-only
// Local Attention Softmax: Apply softmax with local window masking (f64 canonical)
// Pass 2 of local attention (reuses attention matmul and apply passes)
//
// **Local Window**: Position i can only attend to positions within window
// window[i] = [max(0, i - half_window), min(seq_len, i + half_window + 1)]
// mask[i,j] = -inf if j outside window, else 0
//
// Input: attention scores [batch, heads, seq, seq]
// Output: attention weights [batch, heads, seq, seq] (with local window masking)

struct Params {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    window_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f64>;
@group(0) @binding(1) var<storage, read_write> weights: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.seq_len;

    if (idx >= total) {
        return;
    }

    // Decompose index
    let b = idx / (params.num_heads * params.seq_len);
    let h = (idx % (params.num_heads * params.seq_len)) / params.seq_len;
    let i = idx % params.seq_len; // query position

    let base_idx = b * params.num_heads * params.seq_len * params.seq_len +
                   h * params.seq_len * params.seq_len +
                   i * params.seq_len;

    // Compute window bounds
    let half_window = params.window_size / 2u;
    var window_start = 0u;
    if (i >= half_window) {
        window_start = i - half_window;
    }
    var window_end = params.seq_len;
    if (i + half_window + 1u < params.seq_len) {
        window_end = i + half_window + 1u;
    }

    // Find max score (only over valid positions within window)
    var max_score = f64(-1e10);
    for (var j = window_start; j < window_end; j++) {
        let score_idx = base_idx + j;
        max_score = max(max_score, scores[score_idx]);
    }

    // Compute exp and sum (with local window masking)
    var sum = f64(0.0);
    for (var j = 0u; j < params.seq_len; j++) {
        let score_idx = base_idx + j;

        if (j >= window_start && j < window_end) {
            // Valid position within window: apply softmax
            let exp_val = exp_f64(scores[score_idx] - max_score);
            weights[score_idx] = exp_val;
            sum += exp_val;
        } else {
            // Outside window: mask to zero
            weights[score_idx] = f64(0.0);
        }
    }

    // Normalize (only valid positions within window contribute to sum)
    if (sum > f64(0.0)) {
        for (var j = window_start; j < window_end; j++) {
            let score_idx = base_idx + j;
            weights[score_idx] = weights[score_idx] / sum;
        }
    }
}
