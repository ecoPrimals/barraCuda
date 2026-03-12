// SPDX-License-Identifier: AGPL-3.0-only
//! ALiBi Position Encoding - Attention with Linear Biases (f64 canonical)
//!
//! Adds position-dependent bias to attention scores
//! No learned position embeddings needed
//!
//! Input: attention scores [batch, heads, seq, seq]
//! Output: biased scores [batch, heads, seq, seq]
//!
//! Algorithm:
//! - Slope: slope[h] = 2^(-(8*(h+1) / num_heads))
//! - Distance: dist[i,j] = |i - j|
//! - Bias: bias[h,i,j] = -slope[h] * dist[i,j]
//! - Output: scores[b,h,i,j] + bias[h,i,j]
//!
//! Reference: "Train Short, Test Long" (Press et al., 2021)

struct Params {
    batch_size: u32,
    num_heads: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// Compute head-specific slope
fn compute_slope(head: u32, num_heads: u32) -> f64 {
    let exponent = -8.0 * f64(head + 1) / f64(num_heads);
    return pow_f64(f64(2.0), exponent);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads * params.seq_len * params.seq_len;
    
    if (idx >= total) {
        return;
    }

    // Decompose index: [batch, head, i, j]
    let b = idx / (params.num_heads * params.seq_len * params.seq_len);
    let h = (idx % (params.num_heads * params.seq_len * params.seq_len)) / (params.seq_len * params.seq_len);
    let i = (idx % (params.seq_len * params.seq_len)) / params.seq_len;
    let j = idx % params.seq_len;

    // Compute ALiBi bias
    let slope = compute_slope(h, params.num_heads);
    let distance = abs(i32(i) - i32(j)); // Position distance
    let bias = -slope * f64(distance);

    // Add bias to attention score
    output[idx] = scores[idx] + bias;
}
