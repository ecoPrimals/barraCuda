// SPDX-License-Identifier: AGPL-3.0-only
// head_concat.wgsl — Reshape [B, H, S, D] → [B, S, H*D] (f64 canonical)
//
// Inverse of head_split: merges multi-head attention output back into
// a packed representation for the output projection.
//
// Mapping: out[b, s, h*D + d] = in[b, h, s, d]
//
// Used by TensorSession::head_concat() after multi-head attention
// before the output linear projection.

struct HeadConcatParams {
    batch_size: u32,   // B
    seq_len:    u32,   // S
    num_heads:  u32,  // H
    head_dim:   u32,   // D
}

@group(0) @binding(0) var<storage, read>       input:  array<f64>;  // [B, H, S, D]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, S, H*D]
@group(0) @binding(2) var<uniform>             params: HeadConcatParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let B  = params.batch_size;
    let S  = params.seq_len;
    let H  = params.num_heads;
    let D  = params.head_dim;
    let total = B * H * S * D;

    let idx = gid.x;
    if idx >= total { return; }

    // Source index in [B, H, S, D]
    let b = idx / (H * S * D);
    let h = (idx / (S * D)) % H;
    let s = (idx / D) % S;
    let d = idx % D;

    // Destination index in [B, S, H*D]
    let dst_idx = b * S * (H * D) + s * (H * D) + h * D + d;
    output[dst_idx] = input[idx];
}
