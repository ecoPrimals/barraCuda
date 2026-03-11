// SPDX-License-Identifier: AGPL-3.0-only
// msa_attention_df64.wgsl — Multiple sequence alignment attention (row/column)
//
// Combined row+column: score = Q*K^T / sqrt(d)
// Row: for (s,i,j) score[s,i,j] = dot(q[s,i,:], k[s,j,:])/sqrt(d)
// Column: for (n,i,j) score[n,i,j] = dot(q[i,n,:], k[j,n,:])/sqrt(d)
// This shader does row attention. Layout: q[S*N*D], k[S*N*D], out[S*N*N]
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct MsaAttnParams {
    s: u32,
    n: u32,
    d: u32,
}

@group(0) @binding(0) var<storage, read> q: array<vec2<f32>>;  // [S*N*D] as DF64
@group(0) @binding(1) var<storage, read> k: array<vec2<f32>>;  // [S*N*D] as DF64
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;  // [S*N*N] as DF64
@group(0) @binding(3) var<uniform> params: MsaAttnParams;

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

    let q_base = (s * N + i) * D;
    let k_base = (s * N + j) * D;

    var dot = df64_from_f32(0.0);
    for (var d = 0u; d < D; d = d + 1u) {
        let qv = Df64(q[q_base + d].x, q[q_base + d].y);
        let kv = Df64(k[k_base + d].x, k[k_base + d].y);
        dot = df64_add(dot, df64_mul(qv, kv));
    }

    let scale = sqrt_df64(df64_from_f32(f32(D)));
    let score = df64_div(dot, scale);
    out[idx] = vec2<f32>(score.hi, score.lo);
}
