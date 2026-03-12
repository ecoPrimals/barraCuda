// SPDX-License-Identifier: AGPL-3.0-only
// head_split.wgsl — Reshape [B, S, H*D] → [B, H, S, D] (f64 canonical)
//
// Converts a packed QKV projection output (batch-first, sequence-first layout)
// into multi-head layout (batch, head, sequence, head_dim).
//
// Mapping: out[b, h, s, d] = in[b, s, h*D + d]
//
// Used by TensorSession::head_split() to prepare Q/K/V for attention.

struct HeadSplitParams {
    batch_size: u32,   // B
    seq_len:    u32,   // S
    num_heads:  u32,  // H
    head_dim:   u32,   // D   (H * D = total_dim)
}

@group(0) @binding(0) var<storage, read>       input:  array<f64>;  // [B, S, H*D]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, H, S, D]
@group(0) @binding(2) var<uniform>             params: HeadSplitParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One thread per output element
    let B  = params.batch_size;
    let S  = params.seq_len;
    let H  = params.num_heads;
    let D  = params.head_dim;
    let total = B * H * S * D;

    let idx = gid.x;
    if idx >= total { return; }

    // Decode [B, H, S, D] index
    let b = idx / (H * S * D);
    let h = (idx / (S * D)) % H;
    let s = (idx / D) % S;
    let d = idx % D;

    // Map to [B, S, H*D] source
    let src_idx = b * S * (H * D) + s * (H * D) + h * D + d;
    output[idx] = input[src_idx];
}
