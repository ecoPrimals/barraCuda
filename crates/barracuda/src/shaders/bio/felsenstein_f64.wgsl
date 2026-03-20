// SPDX-License-Identifier: AGPL-3.0-or-later
// felsenstein_f64.wgsl — Felsenstein Pruning Phylogenetic Likelihood (f64)
//
// Computes per-site conditional likelihoods for maximum-likelihood
// phylogenetics using Felsenstein's (1981) pruning algorithm.
//
// Level-order parallelism: one dispatch per tree level (bottom-up).
// Each thread handles one (site, node_in_level) pair.
//
// For each internal node n with children l (left) and r (right):
//   L[n][site][s] = P_l[s][·] · L[l][site][·]  ×  P_r[s][·] · L[r][site][·]
//
// where P_x is the 4×4 (DNA) or 20×20 (protein) transition probability
// matrix for branch length t_x, pre-computed on CPU via matrix exponentiation.
//
// Tip nodes are initialised by the Rust wrapper:
//   L[leaf][site][observed_state] = 1.0, all others = 0.0
//
// The root log-likelihood is computed by a final separate reduction pass:
//   log L = Σ_sites  log( Σ_s  π_s × L[root][site][s] )
//
// Use `ops::logsumexp_wgsl::LogsumexpWgsl` for the final summation.
//
// Absorbed from wetSpring handoff §Shader Design 3 (Feb 2026).

struct FelsensteinParams {
    n_sites:           u32,   // number of alignment sites
    n_nodes_this_level: u32,  // nodes being processed in this dispatch
    n_states:          u32,   // 4 for DNA, 20 for protein
    n_nodes_total:     u32,   // total nodes in the tree (for buffer indexing)
}

// Flat index into likelihoods buffer: [node][site][state]
fn lik_idx(node: u32, site: u32, state: u32, n_sites: u32, n_states: u32) -> u32 {
    return node * n_sites * n_states + site * n_states + state;
}

// Flat index into transition_probs buffer: [node][s][j]
// One 4×4 matrix per node (indexed by node, NOT branch — stored per-child-node).
fn tp_idx(node: u32, s: u32, j: u32, n_states: u32) -> u32 {
    return node * n_states * n_states + s * n_states + j;
}

// Note: FelsensteinParams contains only u32 fields, so var<uniform> is valid.
@group(0) @binding(0) var<uniform>            params:           FelsensteinParams;  // all u32
@group(0) @binding(1) var<storage, read>      node_ids:         array<u32>;  // [n_nodes_this_level] global node indices
@group(0) @binding(2) var<storage, read>      left_child:       array<i32>;  // [n_nodes_total]  <0 = leaf
@group(0) @binding(3) var<storage, read>      right_child:      array<i32>;  // [n_nodes_total]
@group(0) @binding(4) var<storage, read>      transition_probs: array<f64>;  // [n_nodes_total × n_states × n_states]
@group(0) @binding(5) var<storage, read_write> likelihoods:     array<f64>;  // [n_nodes_total × n_sites × n_states]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread    = gid.x;
    let n_sites   = params.n_sites;
    let n_states  = params.n_states;
    let n_level   = params.n_nodes_this_level;

    // Decompose thread into (level_node_idx, site)
    let level_node_idx = thread / n_sites;
    let site           = thread % n_sites;

    if (level_node_idx >= n_level) { return; }

    let node  = node_ids[level_node_idx];
    let left  = left_child[node];
    let right = right_child[node];

    // Leaf nodes are pre-initialised; only process internal nodes.
    if (left < 0) { return; }

    let l_node = u32(left);
    let r_node = u32(right);

    for (var s = 0u; s < n_states; s++) {
        // Left child contribution: Σ_j P_left[s][j] × L[left][site][j]
        // Explicit f64 type annotation — WGSL abstract float 0.0 resolves to f32 otherwise.
        var sum_l: f64 = f64(0.0);
        for (var j = 0u; j < n_states; j++) {
            sum_l = sum_l + transition_probs[tp_idx(l_node, s, j, n_states)]
                          * likelihoods[lik_idx(l_node, site, j, n_sites, n_states)];
        }

        // Right child contribution
        var sum_r: f64 = f64(0.0);
        for (var k = 0u; k < n_states; k++) {
            sum_r = sum_r + transition_probs[tp_idx(r_node, s, k, n_states)]
                          * likelihoods[lik_idx(r_node, site, k, n_sites, n_states)];
        }

        likelihoods[lik_idx(node, site, s, n_sites, n_states)] = sum_l * sum_r;
    }
}
