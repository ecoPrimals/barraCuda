// SPDX-License-Identifier: AGPL-3.0-or-later
// Multi-Head Attention: Input Projection with Head Splitting
// Projects Q/K/V through weight matrix and splits into heads
//
// Input: [batch, seq_len, d_model]
// Weight: [d_model, d_model]
// Output: [batch, num_heads, seq_len, head_dim]
//
// Algorithm: output[b,h,s,d] = sum_i(input[b,s,i] * weight[i, h*head_dim + d])

struct Params {
    batch_size: u32,
    seq_len: u32,
    d_model: u32,
    num_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;   // [B, S, D]
@group(0) @binding(1) var<storage, read> weight: array<f64>;  // [D, D]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, H, S, D/H]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x; // batch
    let h = global_id.y; // head
    let s = global_id.z; // seq position

    if (b >= params.batch_size || h >= params.num_heads || s >= params.seq_len) {
        return;
    }

    // Compute projection for this head
    for (var d = 0u; d < params.head_dim; d++) {
        var sum = 0.0;
        
        // Matrix multiply: input[b,s,:] @ weight[:,h*head_dim + d]
        for (var i = 0u; i < params.d_model; i++) {
            let input_idx = b * params.seq_len * params.d_model + s * params.d_model + i;
            let weight_idx = i * params.d_model + (h * params.head_dim + d);
            sum += input[input_idx] * weight[weight_idx];
        }

        // Write to output: [b, h, s, d]
        let out_idx = b * params.num_heads * params.seq_len * params.head_dim +
                      h * params.seq_len * params.head_dim +
                      s * params.head_dim +
                      d;
        output[out_idx] = sum;
    }
}
