// SPDX-License-Identifier: AGPL-3.0-or-later
//
// ensemble_average_f64.wgsl — Ensemble averaging (AlphaFold2)
//
// out[i,d] = (1/M) * Σ_m positions[m,i,d]
// Averages positions over M models for each atom and dimension.
//
// Bindings: @0 positions[M*N*D], @1 out[N*D], @2 uniform{n_models, n_atoms, n_dims}

struct EnsembleAverageParams {
    n_models: u32,
    n_atoms: u32,
    n_dims: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       positions: array<f64>;  // [M*N*D]
@group(0) @binding(1) var<storage, read_write> out: array<f64>;       // [N*D]
@group(0) @binding(2) var<uniform>             params: EnsembleAverageParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let M = params.n_models;
    let N = params.n_atoms;
    let D = params.n_dims;

    let idx = gid.x;
    if idx >= N * D { return; }

    let d = idx % D;
    let i = idx / D;

    var sum_val = f64(0.0);
    for (var m = 0u; m < M; m = m + 1u) {
        let pos_idx = m * N * D + i * D + d;
        sum_val += positions[pos_idx];
    }

    out[idx] = (1.0 / f64(M)) * sum_val;
}
