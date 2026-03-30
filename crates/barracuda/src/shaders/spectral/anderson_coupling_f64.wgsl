// SPDX-License-Identifier: AGPL-3.0-or-later
//
// anderson_coupling_f64.wgsl — Anderson disorder coupling matrix
//
// Computes sparse coupling elements for the Anderson Hamiltonian:
//   H = -t*Δ + V
// on a d-dimensional cubic lattice with nearest-neighbor hopping and periodic BC.
//
// For 1D: H_ij = V_i*δ_ij - t*(δ_{i,j+1} + δ_{i,j-1})
//
// Each thread handles one site, writing its diagonal element and 2*D off-diagonal
// hopping values. Layout: out_off_diag[site_idx * 2*D + dir] = -t for each neighbor.
//
// Provenance: airSpring → ToadStool absorption

struct AndersonParams {
    n: u32,           // Total number of sites
    dim: u32,         // Lattice dimensionality
    extent: u32,      // Sites per dimension (L where N = L^d for hypercube)
    _pad: u32,
    hopping_t: f64,    // Hopping amplitude t
}

@group(0) @binding(0) var<storage, read> potential: array<f64>;           // [N]
@group(0) @binding(1) var<storage, read_write> out_diagonal: array<f64>;  // [N]
@group(0) @binding(2) var<storage, read_write> out_off_diag: array<f64>;  // [2*D*N]
@group(0) @binding(3) var<uniform> params: AndersonParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let t = params.hopping_t;
    let dim = params.dim;
    let n = params.n;

    // Diagonal: H_ii = V_i
    out_diagonal[i] = potential[i];

    // Off-diagonal: -t for each of 2*D neighbors (periodic BC)
    // Layout: for each dimension d, store [negative_dir, positive_dir] coupling = -t
    let off_base = i * 2u * dim;
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        out_off_diag[off_base + 2u * d] = -t;
        out_off_diag[off_base + 2u * d + 1u] = -t;
    }
}
