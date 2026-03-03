// SPDX-License-Identifier: AGPL-3.0-only
//
// symmetrize_f64.wgsl — Symmetrize a square matrix: out[i,j] = (A[i,j] + A[j,i]) / 2 (f64 canonical)
//
// Provenance: neuralSpring baseCamp V18 handoff (Feb 2026)
// Use case: adjacency matrices, covariance matrices, Hessians

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn symmetrize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    let total = n * n;
    if (idx >= total) {
        return;
    }

    let i = idx / n;
    let j = idx % n;
    let ij = i * n + j;
    let ji = j * n + i;

    output[ij] = (input[ij] + input[ji]) * f64(0.5);
}
