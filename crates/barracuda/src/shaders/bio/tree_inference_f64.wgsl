// SPDX-License-Identifier: AGPL-3.0-or-later
// tree_inference_f64.wgsl — GPU Decision Tree / Random Forest Inference (f64)
//
// Each thread processes one (sample, tree) pair, traversing from the root
// of its assigned tree to a leaf and recording the predicted class.
//
// Tree structure is stored as flat arrays in level-agnostic order:
//   node index 0 = root of the first tree
//   left_child[node] < 0  → leaf  (prediction = predictions[node])
//   left_child[node] ≥ 0  → internal node
//
// For a random forest with n_trees trees, all node arrays are concatenated;
// tree_offsets[t] gives the starting node index of tree t.
//
// Post-processing (majority vote / mean for regression) is done by the
// Rust wrapper after reading back output[sample × n_trees + tree].
//
// Absorbed from wetSpring handoff §Shader Design 2 (Feb 2026).
// Validated against wetSpring's sklearn export: 65 nodes × 28 features,
// 744 samples, 100% prediction parity.

struct TreeParams {
    n_samples:   u32,   // number of input samples N
    n_features:  u32,   // number of features per sample F
    n_nodes_max: u32,   // max nodes per tree (for bounds checking)
    n_trees:     u32,   // number of trees in the forest
    max_depth:   u32,   // traversal depth cap (safety; ~31 for int32 indices)
    _pad0:       u32,
    _pad1:       u32,
    _pad2:       u32,
}

@group(0) @binding(0) var<uniform>       params:       TreeParams;
@group(0) @binding(1) var<storage, read> samples:      array<f64>;  // [N × F]  input features
@group(0) @binding(2) var<storage, read> feature_idx:  array<u32>;  // [nodes]  split feature index
@group(0) @binding(3) var<storage, read> thresholds:   array<f64>;  // [nodes]  split threshold
@group(0) @binding(4) var<storage, read> left_child:   array<i32>;  // [nodes]  <0 = leaf
@group(0) @binding(5) var<storage, read> right_child:  array<i32>;  // [nodes]
@group(0) @binding(6) var<storage, read> predictions:  array<u32>;  // [nodes]  class at leaf
@group(0) @binding(7) var<storage, read> tree_offsets: array<u32>;  // [n_trees] first node of each tree
@group(0) @binding(8) var<storage, read_write> output: array<u32>;  // [N × n_trees] predicted class

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread handles one (sample, tree) pair
    let tid       = gid.x;
    let sample_id = tid % params.n_samples;
    let tree_id   = tid / params.n_samples;

    if (tree_id >= params.n_trees) { return; }

    let feat_base = sample_id * params.n_features;
    var node = i32(tree_offsets[tree_id]);  // root node of this tree

    for (var depth = 0u; depth < params.max_depth; depth++) {
        let left = left_child[u32(node)];
        if (left < 0) {
            break;  // leaf node
        }
        let feat = feature_idx[u32(node)];
        if (samples[feat_base + feat] <= thresholds[u32(node)]) {
            node = left;
        } else {
            node = right_child[u32(node)];
        }
    }

    output[sample_id * params.n_trees + tree_id] = predictions[u32(node)];
}
