// SPDX-License-Identifier: AGPL-3.0-or-later
// TransE Knowledge Graph Triple Scoring (f64)
//
// score(h, r, t) = -‖h + r - t‖₂
//
// One thread per triple. Gathers entity/relation embeddings by index,
// computes element-wise (h + r - t), reduces to L2 norm, negates.

struct TranseParams {
    n_triples: u32,
    dim: u32,       // embedding dimension
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       entities:  array<f64>;  // [n_entities, dim]
@group(0) @binding(1) var<storage, read>       relations: array<f64>;  // [n_relations, dim]
@group(0) @binding(2) var<storage, read>       heads:     array<u32>;  // [n_triples]
@group(0) @binding(3) var<storage, read>       rels:      array<u32>;  // [n_triples]
@group(0) @binding(4) var<storage, read>       tails:     array<u32>;  // [n_triples]
@group(0) @binding(5) var<storage, read_write> scores:    array<f64>;  // [n_triples]
@group(0) @binding(6) var<uniform>             params:    TranseParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_triples) { return; }

    let h_id = heads[idx];
    let r_id = rels[idx];
    let t_id = tails[idx];

    let h_base = h_id * params.dim;
    let r_base = r_id * params.dim;
    let t_base = t_id * params.dim;

    var sum_sq: f64 = 0.0;
    for (var d = 0u; d < params.dim; d = d + 1u) {
        let diff = entities[h_base + d] + relations[r_base + d] - entities[t_base + d];
        sum_sq = sum_sq + diff * diff;
    }

    scores[idx] = -sqrt(sum_sq);
}
