// SPDX-License-Identifier: AGPL-3.0-or-later
// unifrac_propagate.wgsl — GPU UniFrac tree propagation (f64)
//
// wetSpring absorption: phylogenetic distance computation.
//
// Bottom-up propagation of sample abundances through a phylogenetic tree
// stored in CSR (flat) format. Two entry points:
//   leaf_init: copy sample matrix into leaf node slots
//   propagate_level: sum child contributions weighted by branch length
//
// Multi-pass: dispatch leaf_init once, then propagate_level per tree level
// (bottom-up).
//
// CPU reference: wetspring_barracuda::bio::unifrac::unweighted_unifrac
//
// Bindings:
//   0: config     uniform { n_nodes, n_samples, n_leaves, _pad }
//   1: parent     [n_nodes] i32 — parent index (-1 = root)
//   2: branch_len [n_nodes] f64 — branch lengths
//   3: sample_mat [n_leaves × n_samples] f64 — leaf abundances
//   4: node_sums  [n_nodes × n_samples] f64 — propagated sums (read_write)

struct UniFracConfig {
    n_nodes:   u32,
    n_samples: u32,
    n_leaves:  u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             config:     UniFracConfig;
@group(0) @binding(1) var<storage, read>       parent:     array<i32>;
@group(0) @binding(2) var<storage, read>       branch_len: array<f64>;
@group(0) @binding(3) var<storage, read>       sample_mat: array<f64>;
@group(0) @binding(4) var<storage, read_write> node_sums:  array<f64>;

@compute @workgroup_size(64)
fn unifrac_leaf_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf = gid.x;
    if leaf >= config.n_leaves {
        return;
    }

    for (var s: u32 = 0u; s < config.n_samples; s = s + 1u) {
        let src = leaf * config.n_samples + s;
        let dst = leaf * config.n_samples + s;
        node_sums[dst] = sample_mat[src];
    }
}

@compute @workgroup_size(64)
fn unifrac_propagate_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= config.n_nodes {
        return;
    }

    let p = parent[node];
    if p < 0 {
        return;
    }

    let bl = branch_len[node];
    let p_u = u32(p);
    for (var s: u32 = 0u; s < config.n_samples; s = s + 1u) {
        let child_val = node_sums[node * config.n_samples + s] * bl;
        node_sums[p_u * config.n_samples + s] = node_sums[p_u * config.n_samples + s] + child_val;
    }
}
