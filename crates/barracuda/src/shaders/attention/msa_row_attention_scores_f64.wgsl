// SPDX-License-Identifier: AGPL-3.0-or-later
//
// msa_row_attention_scores_f64.wgsl — AlphaFold2 MSA row-wise attention scores
//
// Computes: score = Q*K^T / sqrt(d) for each row across sequences.
// Layout: q[S*N*D], k[S*N*D], out[S*N*N]
// For each sequence s and residue pair (i,j), score[s,i,j] = dot(q[s,i,:], k[s,j,:]) / sqrt(d).
//
// Bindings: @0 q[S*N*D], @1 k[S*N*D], @2 out[S*N*N], @3 uniform{s: u32, n: u32, d: u32}

struct MsaRowParams {
    s: u32,
    n: u32,
    d: u32,
}

@group(0) @binding(0) var<storage, read>       q: array<f64>;     // [S*N*D]
@group(0) @binding(1) var<storage, read>       k: array<f64>;     // [S*N*D]
@group(0) @binding(2) var<storage, read_write> out: array<f64>;   // [S*N*N]
@group(0) @binding(3) var<uniform>             params: MsaRowParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let S = params.s;
    let N = params.n;
    let D = params.d;
    let total = S * N * N;
    let idx = gid.x;
    if idx >= total { return; }

    let snn = idx / N;
    let j = idx % N;
    let i = snn % N;
    let s = snn / N;

    if s >= S { return; }

    let q_base = s * N * D + i * D;
    let k_base = s * N * D + j * D;

    var dot = f64(0.0);
    for (var d = 0u; d < D; d = d + 1u) {
        dot += q[q_base + d] * k[k_base + d];
    }

    let scale = sqrt_f64(f64(D));
    out[idx] = dot / scale;
}
