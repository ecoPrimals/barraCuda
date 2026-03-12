// SPDX-License-Identifier: AGPL-3.0-only
// Cross Attention Softmax: Apply softmax to cross-attention scores (f64 canonical)
// Pass 2 of cross-attention
//
// Scores: [batch, heads, decoder_seq, encoder_seq]
// Weights: [batch, heads, decoder_seq, encoder_seq] (after softmax)
//
// For each decoder position i, softmax over all encoder positions j

struct Params {
    batch_size: u32,
    num_heads: u32,
    decoder_seq: u32,
    encoder_seq: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f64>;
@group(0) @binding(1) var<storage, read_write> weights: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.decoder_seq;

    if (idx >= total) {
        return;
    }

    // Decompose: [batch, head, decoder_pos]
    let b = idx / (params.num_heads * params.decoder_seq);
    let h = (idx % (params.num_heads * params.decoder_seq)) / params.decoder_seq;
    let i = idx % params.decoder_seq; // decoder position

    let base_idx = b * params.num_heads * params.decoder_seq * params.encoder_seq +
                   h * params.decoder_seq * params.encoder_seq +
                   i * params.encoder_seq;

    // Find max score over all encoder positions
    var max_score = f64(-1e10);
    for (var j = 0u; j < params.encoder_seq; j++) {
        max_score = max(max_score, scores[base_idx + j]);
    }

    // Compute exp and sum
    var sum = f64(0.0);
    for (var j = 0u; j < params.encoder_seq; j++) {
        let exp_val = exp_f64(scores[base_idx + j] - max_score);
        weights[base_idx + j] = exp_val;
        sum += exp_val;
    }

    // Normalize
    for (var j = 0u; j < params.encoder_seq; j++) {
        weights[base_idx + j] = weights[base_idx + j] / sum;
    }
}
