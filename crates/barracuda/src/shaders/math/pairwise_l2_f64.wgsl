// SPDX-License-Identifier: AGPL-3.0-or-later
// pairwise_l2_f64.wgsl — Pairwise Euclidean distance matrix (f64 canonical)
//
// neuralSpring absorption: novelty search / MODES diversity metric.
//
// Computes upper-triangle pairwise L2 distances for N vectors of dimension D:
//   output[pair] = sqrt( Σ_d (input[i*D+d] - input[j*D+d])² )
//
// Pair indexing (upper triangle, no diagonal):
//   pair_idx ∈ [0, N*(N-1)/2)
//   Decoded via: i = N - 2 - floor(sqrt(-8*pair+4*N*(N-1)-7)/2 - 0.5)
//                j = pair + i + 1 - N*(N-1)/2 + (N-i)*((N-i)-1)/2
//
// Bindings:
//   0: input  [N × D] f64 — row-major feature vectors
//   1: output [N*(N-1)/2] f64 — pairwise L2 distances
//   2: params uniform { n: u32, dim: u32 }

struct PairwiseParams {
    n:   u32,
    dim: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform>             params: PairwiseParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let dim = params.dim;
    let n_pairs = n * (n - 1u) / 2u;
    let pair = global_id.x;
    if (pair >= n_pairs) { return; }

    // Decode pair index → (i, j) with i < j
    let nf = f64(n);
    let pf = f64(pair);
    let disc = sqrt(max(0.0, -8.0 * pf + 4.0 * nf * (nf - 1.0) - 7.0));
    let i = u32(nf - 2.0 - floor(disc * 0.5 - 0.5));
    let j = pair + i + 1u - n * (n - 1u) / 2u + (n - i) * ((n - i) - 1u) / 2u;

    // Accumulate squared differences
    var sum_sq: f64 = 0.0;
    let base_i = i * dim;
    let base_j = j * dim;
    for (var d = 0u; d < dim; d = d + 1u) {
        let diff = input[base_i + d] - input[base_j + d];
        sum_sq = sum_sq + diff * diff;
    }

    output[pair] = sqrt(sum_sq);
}
