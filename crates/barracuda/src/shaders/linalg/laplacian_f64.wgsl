// SPDX-License-Identifier: AGPL-3.0-or-later
//
// laplacian_f64.wgsl — Graph Laplacian: L = D - A (f64 canonical)
//
// D is the degree matrix (diagonal = row sums of A).
// L[i,j] = degree(i) if i==j, else -A[i,j]
//
// Provenance: neuralSpring baseCamp V18 handoff (Feb 2026)
// Use case: spectral graph theory, network analysis, ecology, genomics

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> adjacency: array<f64>;
@group(0) @binding(1) var<storage, read_write> laplacian: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn graph_laplacian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.n;
    let total = n * n;
    if (idx >= total) {
        return;
    }

    let i = idx / n;
    let j = idx % n;

    if (i == j) {
        var degree: f64 = f64(0.0);
        for (var k: u32 = 0u; k < n; k++) {
            degree += adjacency[i * n + k];
        }
        laplacian[idx] = degree - adjacency[idx];
    } else {
        laplacian[idx] = -adjacency[idx];
    }
}
