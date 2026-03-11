// SPDX-License-Identifier: AGPL-3.0-only
//
// band_edges_parallel_f64.wgsl — Band edges from eigensystem results
//
// Finds band edges (min/max eigenvalues) in parallel for N eigensystems.
// Each eigensystem has M eigenvalues sorted ascending.
// Band min = eigenvalue[0], band max = eigenvalue[M-1].
//
// Provenance: groundSpring → ToadStool absorption

enable f64;

struct BandEdgesParams {
    n: u32,   // Number of eigensystems
    m: u32,   // Eigenvalues per system
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> eigenvalues: array<f64>;         // [N*M]
@group(0) @binding(1) var<storage, read_write> out_band_min: array<f64>;  // [N]
@group(0) @binding(2) var<storage, read_write> out_band_max: array<f64>;  // [N]
@group(0) @binding(3) var<uniform> params: BandEdgesParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let m = params.m;
    let base = i * m;

    out_band_min[i] = eigenvalues[base];
    out_band_max[i] = eigenvalues[base + m - 1u];
}
