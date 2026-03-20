// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rotary Position Embedding (RoPE) - GPU implementation (f64 canonical)
//!
//! Applies 2D rotation to query/key embeddings based on position
//! Encodes relative position without absolute embeddings
//!
//! Input: [batch, seq_len, num_heads, head_dim]
//! Output: [batch, seq_len, num_heads, head_dim] (rotated)
//!
//! Algorithm:
//! - Frequency: freq[i] = 1 / (10000^(2i / head_dim))
//! - Theta: theta[pos,i] = pos * freq[i]
//! - Rotation: [x1', x2'] = [cos(theta) -sin(theta)] [x1]
//!                           [sin(theta)  cos(theta)] [x2]
//!
//! Reference: RoFormer (Su et al., 2021)

struct Params {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// Compute frequency for dimension i
fn compute_freq(i: u32, head_dim: u32) -> f64 {
    let exponent = 2.0 * f64(i) / f64(head_dim);
    return f64(1.0) / pow_f64(f64(10000.0), exponent);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.seq_len * params.num_heads * params.half_dim;
    
    if (idx >= total) {
        return;
    }

    // Decompose index: [batch, seq, head, dim_pair]
    let b = idx / (params.seq_len * params.num_heads * params.half_dim);
    let s = (idx % (params.seq_len * params.num_heads * params.half_dim)) / (params.num_heads * params.half_dim);
    let h = (idx % (params.num_heads * params.half_dim)) / params.half_dim;
    let d = idx % params.half_dim;

    // Compute frequency and rotation angle
    let freq = compute_freq(d, params.head_dim);
    let pos = f64(s);
    let theta = pos * freq;
    let cos_val = cos_f64(theta);
    let sin_val = sin_f64(theta);

    // Indices for the pair [x1, x2]
    let idx1 = b * params.seq_len * params.num_heads * params.head_dim +
               s * params.num_heads * params.head_dim +
               h * params.head_dim +
               d;
    let idx2 = idx1 + params.half_dim;

    // Read input pair
    let x1 = input[idx1];
    let x2 = input[idx2];

    // Apply 2D rotation
    // [x1']   [cos  -sin] [x1]
    // [x2'] = [sin   cos] [x2]
    output[idx1] = x1 * cos_val - x2 * sin_val;
    output[idx2] = x1 * sin_val + x2 * cos_val;
}
