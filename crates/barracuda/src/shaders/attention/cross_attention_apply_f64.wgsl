// SPDX-License-Identifier: AGPL-3.0-or-later
// Cross Attention Apply: Apply attention weights to encoder values
// Pass 3 of cross-attention
//
// Weights: [batch, heads, decoder_seq, encoder_seq]
// Value (encoder): [batch, heads, encoder_seq, dim]
// Output (decoder): [batch, heads, decoder_seq, dim]
//
// output[b,h,i,d] = sum_j(weights[b,h,i,j] * value[b,h,j,d])

struct Params {
    batch_size: u32,
    num_heads: u32,
    decoder_seq: u32,
    encoder_seq: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f64>; // [B, H, Dec, Enc]
@group(0) @binding(1) var<storage, read> value: array<f64>;   // [B, H, Enc, D]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, H, Dec, D]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.decoder_seq * params.head_dim;
    
    if (idx >= total) {
        return;
    }

    // Decompose: [batch, head, decoder_pos, dim]
    let b = idx / (params.num_heads * params.decoder_seq * params.head_dim);
    let h = (idx % (params.num_heads * params.decoder_seq * params.head_dim)) / (params.decoder_seq * params.head_dim);
    let i = (idx % (params.decoder_seq * params.head_dim)) / params.head_dim; // decoder pos
    let d = idx % params.head_dim;

    // Compute weighted sum over encoder sequence
    var weighted_sum = 0.0;
    for (var j = 0u; j < params.encoder_seq; j++) {
        let weight_idx = b * params.num_heads * params.decoder_seq * params.encoder_seq +
                        h * params.decoder_seq * params.encoder_seq +
                        i * params.encoder_seq +
                        j;
        let value_idx = b * params.num_heads * params.encoder_seq * params.head_dim +
                       h * params.encoder_seq * params.head_dim +
                       j * params.head_dim +
                       d;
        weighted_sum += weights[weight_idx] * value[value_idx];
    }

    output[idx] = weighted_sum;
}
