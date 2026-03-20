// SPDX-License-Identifier: AGPL-3.0-or-later
// GCNConv - Graph Convolutional Network (f64 canonical)

struct Params {
    num_nodes: u32,
    num_edges: u32,
    in_features: u32,
    out_features: u32,
    add_self_loops: u32,
    normalize: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> node_features: array<f64>;
@group(0) @binding(2) var<storage, read> edge_index: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f64>;
@group(0) @binding(4) var<storage, read> degrees: array<f64>;
@group(0) @binding(5) var<storage, read_write> transformed: array<f64>;
@group(0) @binding(6) var<storage, read_write> output: array<atomic<i32>>;

@compute @workgroup_size(256)
fn transform_features(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node = global_id.x;
    if (node >= params.num_nodes) {
        return;
    }

    for (var out_f = 0u; out_f < params.out_features; out_f = out_f + 1u) {
        var sum: f64 = 0.0;
        for (var in_f = 0u; in_f < params.in_features; in_f = in_f + 1u) {
            let feat_idx = node * params.in_features + in_f;
            let weight_idx = in_f * params.out_features + out_f;
            sum += node_features[feat_idx] * weights[weight_idx];
        }
        transformed[node * params.out_features + out_f] = sum;
    }
}

@compute @workgroup_size(256)
fn aggregate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge = global_id.x;
    if (edge >= params.num_edges) {
        return;
    }

    let src = edge_index[edge * 2u];
    let dst = edge_index[edge * 2u + 1u];

    var norm: f64 = 1.0;
    if (params.normalize != 0u) {
        norm = 1.0 / (degrees[src] * degrees[dst]);
    }

    for (var f = 0u; f < params.out_features; f = f + 1u) {
        let val = norm * transformed[src * params.out_features + f];
        atomicAdd(&output[dst * params.out_features + f], bitcast<i32>(f32(val)));
    }
}

@compute @workgroup_size(256)
fn add_self_loops(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node = global_id.x;
    if (node >= params.num_nodes) {
        return;
    }

    if (params.add_self_loops != 0u) {
        for (var f = 0u; f < params.out_features; f = f + 1u) {
            let idx = node * params.out_features + f;
            atomicAdd(&output[idx], bitcast<i32>(f32(transformed[idx])));
        }
    }
}
