// SPDX-License-Identifier: AGPL-3.0-only
// rf_batch_inference.wgsl — Batch Random Forest Inference (SoA layout)
//
// One thread per (sample, tree) pair. Each thread traverses one tree
// for one sample. Results stored in a flat [n_samples × n_trees] array,
// then reduced by CPU for majority vote.
//
// SoA layout avoids bitcast — thresholds stored as native f64.
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct RfParams {
    n_samples:    u32,
    n_trees:      u32,
    n_nodes_max:  u32,
    n_features:   u32,
}

@group(0) @binding(0) var<uniform>             params:        RfParams;
// feature < 0 means leaf
@group(0) @binding(1) var<storage, read>       node_features: array<i32>;  // [n_trees × n_nodes_max]
@group(0) @binding(2) var<storage, read>       node_thresh:   array<f64>;  // [n_trees × n_nodes_max]
// Children: [n_trees × n_nodes_max × 2]. For leaves: left = predicted class.
@group(0) @binding(3) var<storage, read>       node_children: array<i32>;
@group(0) @binding(4) var<storage, read>       features:      array<f64>;  // [n_samples × n_features]
@group(0) @binding(5) var<storage, read_write> predictions:   array<u32>;  // [n_samples × n_trees]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_idx = gid.x;
    let total = params.n_samples * params.n_trees;
    if global_idx >= total { return; }

    let sample_idx = global_idx / params.n_trees;
    let tree_idx = global_idx % params.n_trees;

    let feat_base = sample_idx * params.n_features;
    let tree_base = tree_idx * params.n_nodes_max;
    var node_idx: u32 = 0u;

    for (var step: u32 = 0u; step < params.n_nodes_max; step = step + 1u) {
        let noff = tree_base + node_idx;
        let feature_id = node_features[noff];

        if feature_id < 0 {
            let child_off = (tree_base + node_idx) * 2u;
            predictions[global_idx] = u32(node_children[child_off]);
            return;
        }

        let threshold = node_thresh[noff];
        let feat_val = features[feat_base + u32(feature_id)];
        let child_off = (tree_base + node_idx) * 2u;

        if feat_val <= threshold {
            node_idx = u32(node_children[child_off]);
        } else {
            node_idx = u32(node_children[child_off + 1u]);
        }
    }

    predictions[global_idx] = 0u;
}
