// SPDX-License-Identifier: AGPL-3.0-or-later
// Multi-Head Attention: Output Projection with Head Concatenation (f64 canonical)
// Concatenates heads and projects through output weight matrix
//
// Input: [batch, num_heads, seq_len, head_dim]
// Weight: [d_model, d_model]
// Output: [batch, seq_len, d_model]
//
// Algorithm:
// 1. Reshape [B,H,S,D/H] → [B,S,D] (concatenate heads)
// 2. Project: output[b,s,i] = sum_j(concat[b,s,j] * weight[j,i])

struct Params {
    batch_size: u32,
    seq_len: u32,
    d_model: u32,
    num_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;   // [B, H, S, D/H]
@group(0) @binding(1) var<storage, read> weight: array<f64>;  // [D, D]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, S, D]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x; // batch
    let s = global_id.y; // seq position
    let out_dim = global_id.z; // output dimension

    if (b >= params.batch_size || s >= params.seq_len || out_dim >= params.d_model) {
        return;
    }

    var sum = 0.0;

    // Concatenate heads and project through output matrix
    for (var h = 0u; h < params.num_heads; h++) {
        for (var d = 0u; d < params.head_dim; d++) {
            // Read from input: [b, h, s, d]
            let input_idx = b * params.num_heads * params.seq_len * params.head_dim +
                           h * params.seq_len * params.head_dim +
                           s * params.head_dim +
                           d;

            // Concatenated index in flat array [b, s, h*head_dim + d]
            let concat_dim = h * params.head_dim + d;

            // Weight index: [concat_dim, out_dim]
            let weight_idx = concat_dim * params.d_model + out_dim;

            sum += input[input_idx] * weight[weight_idx];
        }
    }

    // Write output: [b, s, out_dim]
    let out_idx = b * params.seq_len * params.d_model + s * params.d_model + out_dim;
    output[out_idx] = sum;
}
