// SPDX-License-Identifier: AGPL-3.0-or-later
// pair_representation_df64.wgsl — Pair features update (outer product mean)
//
// out[i,j,c] = (1/S) * sum_s a[s,i,c] * b[s,j,c]
// AlphaFold2 Evoformer outer product mean over MSA sequences.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct PairRepParams {
    s: u32,
    n: u32,
    c: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<vec2<f32>>;  // [S*N*C] as DF64
@group(0) @binding(1) var<storage, read> b: array<vec2<f32>>;  // [S*N*C] as DF64
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;  // [N*N*C] as DF64
@group(0) @binding(3) var<uniform> params: PairRepParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let S = params.s;
    let N = params.n;
    let C = params.c;

    let idx = gid.x;
    if idx >= N * N * C { return; }

    let c = idx % C;
    let j = (idx / C) % N;
    let i = idx / (N * C);

    var sum_val = df64_from_f32(0.0);
    for (var s = 0u; s < S; s = s + 1u) {
        let a_idx = s * N * C + i * C + c;
        let b_idx = s * N * C + j * C + c;
        let av = Df64(a[a_idx].x, a[a_idx].y);
        let bv = Df64(b[b_idx].x, b[b_idx].y);
        sum_val = df64_add(sum_val, df64_mul(av, bv));
    }

    let inv_s = df64_div(df64_from_f32(1.0), df64_from_f32(f32(S)));
    let result = df64_mul(inv_s, sum_val);
    out[idx] = vec2<f32>(result.hi, result.lo);
}
