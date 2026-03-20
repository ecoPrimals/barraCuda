// SPDX-License-Identifier: AGPL-3.0-or-later
// sdpa_scores_f64.wgsl — QK^T / sqrt(d_k) — Pass 1 of 3-pass SDPA (f64 canonical)
//
// Computes scaled dot-product scores for each (batch, head, query_pos) triple.
// Output layout: scores[batch, head, q_pos, k_pos] (row-major)
// Supports cross-attention: Q seq_len may differ from K/V seq_len.
//
// Used by TensorSession::attention() as the first of 3 sequential passes:
//   1. sdpa_scores  → scores[B, H, Sq, Skv]
//   2. attention_softmax → weights[B, H, Sq, Skv]
//   3. attention_apply   → output[B, H, Sq, D]

struct AttentionParams {
    batch_size: u32,
    num_heads:  u32,
    q_seq_len:  u32,
    kv_seq_len: u32,
    head_dim:   u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<storage, read>       query:  array<f64>;  // [B, H, Sq, D]
@group(0) @binding(1) var<storage, read>       key:    array<f64>;  // [B, H, Skv, D]
@group(0) @binding(2) var<storage, read_write> scores: array<f64>;  // [B, H, Sq, Skv]
@group(0) @binding(3) var<uniform>             params: AttentionParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.batch_size * params.num_heads * params.q_seq_len * params.kv_seq_len;
    let idx   = gid.x;
    if idx >= total { return; }

    let bh     = idx / (params.q_seq_len * params.kv_seq_len);
    let rem    = idx % (params.q_seq_len * params.kv_seq_len);
    let q_pos  = rem / params.kv_seq_len;
    let k_pos  = rem % params.kv_seq_len;

    let b    = bh / params.num_heads;
    let h    = bh % params.num_heads;

    if b >= params.batch_size { return; }

    let D     = params.head_dim;
    let Sq    = params.q_seq_len;
    let Skv   = params.kv_seq_len;
    let H     = params.num_heads;

    let q_base = b * H * Sq * D + h * Sq * D + q_pos * D;
    let k_base = b * H * Skv * D + h * Skv * D + k_pos * D;

    var dot = f64(0.0);
    for (var d = 0u; d < D; d++) {
        dot += query[q_base + d] * key[k_base + d];
    }

    let scale  = sqrt_f64(f64(D));
    let out_idx = b * H * Sq * Skv + h * Sq * Skv + q_pos * Skv + k_pos;
    scores[out_idx] = dot / scale;
}
