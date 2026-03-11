// SPDX-License-Identifier: AGPL-3.0-only
//
// template_embedding_f64.wgsl — AlphaFold2 template stack averaging
//
// out[i,j,c] = (1/T) * Σ_t template[t,i,j,c]
// Averages template representations over T templates for pair representation.
//
// Bindings: @0 templates[T*N*N*C], @1 out[N*N*C], @2 uniform{t, n, c}

enable f64;

struct TemplateEmbeddingParams {
    t: u32,
    n: u32,
    c: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       templates: array<f64>;  // [T*N*N*C]
@group(0) @binding(1) var<storage, read_write> out: array<f64>;       // [N*N*C]
@group(0) @binding(2) var<uniform>             params: TemplateEmbeddingParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T = params.t;
    let N = params.n;
    let C = params.c;

    let idx = gid.x;
    if idx >= N * N * C { return; }

    var sum_val = f64(0.0);
    for (var t = 0u; t < T; t = t + 1u) {
        let t_idx = t * N * N * C + idx;
        sum_val += templates[t_idx];
    }

    out[idx] = (1.0 / f64(T)) * sum_val;
}
