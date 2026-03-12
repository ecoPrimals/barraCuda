// SPDX-License-Identifier: AGPL-3.0-only
// Sparse Attention Softmax: Apply softmax with strided sparse mask (f64 canonical)
// Pass 2 of sparse attention (reuses attention matmul and apply passes)
//
// **Sparse Pattern**: Position i only attends to positions j where j % stride == 0
// This reduces computation for long sequences
// Example stride=4: attend to positions [0, 4, 8, 12, 16, ...]
//
// Input: attention scores [batch, heads, seq, seq]
// Output: attention weights [batch, heads, seq, seq] (with sparse masking)

struct Params {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    stride: u32,
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

    // Find max score (only over strided positions)
    var max_score = f64(-1e10);
    for (var j = 0u; j < params.seq_len; j += params.stride) {
        let score_idx = base_idx + j;
        max_score = max(max_score, scores[score_idx]);
    }

    // Compute exp and sum (with sparse masking)
    var sum = f64(0.0);
    for (var j = 0u; j < params.seq_len; j++) {
        let score_idx = base_idx + j;

        if (j % params.stride == 0u) {
            // Valid strided position: apply softmax
            let exp_val = exp_f64(scores[score_idx] - max_score);
            weights[score_idx] = exp_val;
            sum += exp_val;
        } else {
            // Non-strided position: mask to zero (sparse pattern)
            weights[score_idx] = f64(0.0);
        }
    }

    // Normalize (only strided positions contribute to sum)
    for (var j = 0u; j < params.seq_len; j += params.stride) {
        let score_idx = base_idx + j;
        weights[score_idx] = weights[score_idx] / sum;
    }
}
