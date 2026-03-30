// SPDX-License-Identifier: AGPL-3.0-or-later
//
// triangle_attention_f64.wgsl — AlphaFold2 Evoformer triangle self-attention (starting node)
//
// Computes: attn_ij = softmax_k(b_ij * q_ik * k_jk / sqrt(d)) * v_jk
// For each pair (i,j): score over k = b[i,j] * dot(q[i,k,:], k[j,k,:]) / sqrt(d),
// softmax over k, then weighted sum of v[j,k,:].
//
// Bindings: @0 pair[N*N], @1 q[N*N*D], @2 k[N*N*D], @3 v[N*N*D], @4 out[N*N*D], @5 uniform{n: u32, d: u32}

struct TriangleAttnParams {
    n: u32,
    d: u32,
}

fn softmax_score(x: f64, max_val: f64) -> f64 {
    return exp_f64(x - max_val);
}

@group(0) @binding(0) var<storage, read>       pair: array<f64>;   // [N*N]
@group(0) @binding(1) var<storage, read>       q: array<f64>;     // [N*N*D]
@group(0) @binding(2) var<storage, read>       k: array<f64>;     // [N*N*D]
@group(0) @binding(3) var<storage, read>       v: array<f64>;     // [N*N*D]
@group(0) @binding(4) var<storage, read_write> out: array<f64>;   // [N*N*D]
@group(0) @binding(5) var<uniform>             params: TriangleAttnParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let D = params.d;
    let total = N * N * D;
    let idx = gid.x;
    if idx >= total { return; }

    let nn = idx / D;
    let d = idx % D;
    let i = nn / N;
    let j = nn % N;

    let b_ij = pair[i * N + j];
    let scale = sqrt_f64(f64(D));

    var max_score = f64(-1e30);
    for (var k = 0u; k < N; k = k + 1u) {
        var dot = f64(0.0);
        let q_base = i * N * D + k * D;
        let k_base = j * N * D + k * D;
        for (var dd = 0u; dd < D; dd = dd + 1u) {
            dot += q[q_base + dd] * k[k_base + dd];
        }
        let s = b_ij * dot / scale;
        max_score = max(max_score, s);
    }

    var sum_exp = f64(0.0);
    for (var k = 0u; k < N; k = k + 1u) {
        var dot = f64(0.0);
        let q_base = i * N * D + k * D;
        let k_base = j * N * D + k * D;
        for (var dd = 0u; dd < D; dd = dd + 1u) {
            dot += q[q_base + dd] * k[k_base + dd];
        }
        let s = b_ij * dot / scale;
        sum_exp += softmax_score(s, max_score);
    }

    var weighted = f64(0.0);
    for (var k = 0u; k < N; k = k + 1u) {
        var dot = f64(0.0);
        let q_base = i * N * D + k * D;
        let k_base = j * N * D + k * D;
        for (var dd = 0u; dd < D; dd = dd + 1u) {
            dot += q[q_base + dd] * k[k_base + dd];
        }
        let s = b_ij * dot / scale;
        let w = softmax_score(s, max_score) / sum_exp;
        let v_idx = j * N * D + k * D + d;
        weighted += w * v[v_idx];
    }

    out[idx] = weighted;
}
