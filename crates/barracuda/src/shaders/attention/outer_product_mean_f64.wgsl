// SPDX-License-Identifier: AGPL-3.0-or-later
//
// outer_product_mean_f64.wgsl — AlphaFold2 Evoformer outer product mean
//
// Computes: out[i,j,c] = (1/S) * Σ_s a[s,i,c] * b[s,j,c]
// Projected outer product averaged over MSA sequences for pair representation update.
//
// Bindings: @0 a[S,N,C], @1 b[S,N,C], @2 out[N,N,C], @3 uniform{s,n,c}

enable f64;

struct OpmParams {
    s: u32,
    n: u32,
    c: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a: array<f64>;   // [S, N, C]
@group(0) @binding(1) var<storage, read>       b: array<f64>;   // [S, N, C]
@group(0) @binding(2) var<storage, read_write> out: array<f64>; // [N, N, C]
@group(0) @binding(3) var<uniform>             params: OpmParams;

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

    var sum_val = f64(0.0);
    for (var s = 0u; s < S; s = s + 1u) {
        let a_idx = s * N * C + i * C + c;
        let b_idx = s * N * C + j * C + c;
        sum_val += a[a_idx] * b[b_idx];
    }

    out[idx] = (1.0 / f64(S)) * sum_val;
}
