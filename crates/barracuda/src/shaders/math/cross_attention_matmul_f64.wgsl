// Cross Attention Matrix Multiplication: Compute QK^T scores (f64 canonical)
// Pass 1 of cross-attention (decoder attends to encoder)
//
// Q (decoder): [batch, heads, decoder_seq, dim]
// K (encoder): [batch, heads, encoder_seq, dim]
// Scores: [batch, heads, decoder_seq, encoder_seq]
//
// scores[b,h,i,j] = sum_d(Q[b,h,i,d] * K[b,h,j,d]) / sqrt(dim)

struct Params {
    batch_size: u32,
    num_heads: u32,
    decoder_seq: u32,
    encoder_seq: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f64>;  // [B, H, Dec, D]
@group(0) @binding(1) var<storage, read> key: array<f64>;    // [B, H, Enc, D]
@group(0) @binding(2) var<storage, read_write> scores: array<f64>; // [B, H, Dec, Enc]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.decoder_seq * params.encoder_seq;
    
    if (idx >= total) {
        return;
    }

    // Decompose index
    let b = idx / (params.num_heads * params.decoder_seq * params.encoder_seq);
    let h = (idx % (params.num_heads * params.decoder_seq * params.encoder_seq)) / (params.decoder_seq * params.encoder_seq);
    let i = (idx % (params.decoder_seq * params.encoder_seq)) / params.encoder_seq; // decoder pos
    let j = idx % params.encoder_seq; // encoder pos

    // Compute dot product Q[b,h,i,:] · K[b,h,j,:]
    var score = f64(0.0);
    for (var d = 0u; d < params.head_dim; d++) {
        let q_idx = b * params.num_heads * params.decoder_seq * params.head_dim +
                    h * params.decoder_seq * params.head_dim +
                    i * params.head_dim +
                    d;
        let k_idx = b * params.num_heads * params.encoder_seq * params.head_dim +
                    h * params.encoder_seq * params.head_dim +
                    j * params.head_dim +
                    d;
        score += query[q_idx] * key[k_idx];
    }

    // Scale by sqrt(head_dim)
    let scale = sqrt_f64(f64(params.head_dim));
    scores[idx] = score / scale;
}
