// SPDX-License-Identifier: AGPL-3.0-only
//
// msa_col_attention_scores_f64.wgsl — AlphaFold2 MSA column-wise attention scores
//
// Computes: score = Q*K^T / sqrt(d) for each column across residues.
// Layout: q[S*N*D], k[S*N*D], out[N*S*S]
// For each residue n and sequence pair (i,j), score[n,i,j] = dot(q[i,n,:], k[j,n,:]) / sqrt(d).
//
// Bindings: @0 q[S*N*D], @1 k[S*N*D], @2 out[N*S*S], @3 uniform{s: u32, n: u32, d: u32}

enable f64;

struct MsaColParams {
    s: u32,
    n: u32,
    d: u32,
}

@group(0) @binding(0) var<storage, read>       q: array<f64>;     // [S*N*D]
@group(0) @binding(1) var<storage, read>       k: array<f64>;     // [S*N*D]
@group(0) @binding(2) var<storage, read_write> out: array<f64>;   // [N*S*S]
@group(0) @binding(3) var<uniform>             params: MsaColParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let S = params.s;
    let N = params.n;
    let D = params.d;
    let total = N * S * S;
    let idx = gid.x;
    if idx >= total { return; }

    let nss = idx / S;
    let j = idx % S;
    let i = nss % S;
    let n = nss / S;

    if n >= N { return; }

    let q_base = i * N * D + n * D;
    let k_base = j * N * D + n * D;

    var dot = f64(0.0);
    for (var d = 0u; d < D; d = d + 1u) {
        dot += q[q_base + d] * k[k_base + d];
    }

    let scale = sqrt_f64(f64(D));
    out[idx] = dot / scale;
}
